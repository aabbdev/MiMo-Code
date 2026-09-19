#!/usr/bin/env bun
/**
 * Aggregate MiMoCode debug logs into a prompt-efficiency report.
 *
 * Usage:
 *   bun packages/opencode/script/cache-report.ts <logfile|logdir> [more...]
 *   bun packages/opencode/script/cache-report.ts ~/.local/share/mimocode/log
 *
 * Reads the DEBUG lines emitted by the Phase 0 instrumentation and prints, per
 * log source, the metrics that matter for prompt efficiency:
 *   - context tokens processed per model call (p50 / p90 / max)
 *   - cache-read share of input, and the unbilled (full-price) input
 *   - calls that re-read >50k tokens at full price (rebuild / prune misses)
 *   - which models received cache markers (cache.diagnostics)
 *   - how often the working-set governor capped a result (working-set cap applied)
 *
 * Passing two or more sources prints a side-by-side BEFORE/AFTER comparison, so
 * a change can be judged by numbers rather than by impression.
 *
 * Enable the logs with:  mimo --log-level DEBUG
 */
import { readFileSync, readdirSync, statSync } from "node:fs"
import os from "node:os"
import path from "node:path"

const DEFAULT_LOG_DIR = path.join(os.homedir(), ".local", "share", "mimocode", "log")
const BIG_UNCACHED = 50_000

type Usage = {
  ctx: number
  input: number
  cacheRead: number
  cacheWrite: number
  /** raw provider usage reported cached tokens while the parsed value was 0. */
  sdKDroppedCache: boolean
}

type Source = {
  name: string
  usage: Usage[]
  markers: Map<string, { calls: number; supports: boolean }>
  caps: { count: number; minCap: number; maxCap: number; scopes: Set<string> }
  sizes?: { systemChars: number; toolCount: number; toolsChars: number }
}

function resolveSources(args: string[]): string[] {
  const roots = args.length > 0 ? args : [DEFAULT_LOG_DIR]
  const files: string[] = []
  for (const root of roots) {
    let stat
    try {
      stat = statSync(root)
    } catch {
      console.error(`skip (not found): ${root}`)
      continue
    }
    if (stat.isDirectory()) {
      const entries = readdirSync(root)
        .filter((entry) => entry.endsWith(".log"))
        .sort()
      for (const entry of entries) files.push(path.join(root, entry))
    } else {
      files.push(root)
    }
  }
  // Explicit arguments keep the caller's order so `report before.log after.log`
  // reads BEFORE -> AFTER; only directory listings are sorted.
  return files
}

/** Messages we understand, longest first so a suffix match cannot be shadowed. */
const MESSAGES = ["working-set cap applied", "cache.diagnostics", "request.size", "cache.usage"].sort(
  (a, b) => b.length - a.length,
)

/**
 * One log line is: `<LEVEL> <iso> +<ms>ms k=v k=v ... <message>`.
 * A message may contain spaces, so it is matched by SUFFIX against the known
 * set rather than taken as the last token. Values are whitespace-free
 * (JSON.stringify adds none), so the remaining prefix splits cleanly into `k=v`.
 */
function parseLine(line: string): { message: string; fields: Record<string, string> } | undefined {
  const level = line.match(/^(?:DEBUG|INFO|WARN|ERROR)\s+(.*)$/)
  if (!level) return undefined
  const body = level[1].trimEnd()
  const message = MESSAGES.find((candidate) => body.endsWith(candidate))
  if (!message) return undefined
  const tokens = body.slice(0, body.length - message.length).trim().split(/\s+/)
  const fields: Record<string, string> = {}
  for (const token of tokens) {
    const eq = token.indexOf("=")
    if (eq <= 0) continue
    fields[token.slice(0, eq)] = token.slice(eq + 1)
  }
  return { message, fields }
}

function num(value?: string): number | undefined {
  if (value === undefined) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function parseJson(value?: string): Record<string, unknown> | undefined {
  if (!value) return undefined
  try {
    const parsed = JSON.parse(value)
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : undefined
  } catch {
    return undefined
  }
}

function readSource(file: string): Source {
  const source: Source = { name: path.basename(file), usage: [], markers: new Map(), caps: { count: 0, minCap: Infinity, maxCap: 0, scopes: new Set() } }
  let text: string
  try {
    text = readFileSync(file, "utf8")
  } catch {
    return source
  }
  for (const line of text.split("\n")) {
    if (!line.includes("cache.") && !line.includes("working-set") && !line.includes("request.size")) continue
    const parsed = parseLine(line)
    if (!parsed) continue
    const { message, fields } = parsed

    if (message === "request.size") {
      source.sizes = {
        systemChars: num(fields.systemChars) ?? 0,
        toolCount: num(fields.toolCount) ?? 0,
        toolsChars: num(fields.toolsChars) ?? 0,
      }
      continue
    }

    if (message === "cache.usage") {
      const input = num(fields.parsedInput) ?? 0
      const cacheRead = num(fields.parsedCacheRead) ?? 0
      const cacheWrite = num(fields.parsedCacheWrite) ?? 0
      const raw = parseJson(fields.raw)
      const rawCached = typeof raw?.cached_tokens === "number" ? (raw.cached_tokens as number) : 0
      source.usage.push({ ctx: input + cacheRead + cacheWrite, input, cacheRead, cacheWrite, sdKDroppedCache: cacheRead === 0 && rawCached > 0 })
      continue
    }

    if (message === "cache.diagnostics") {
      const key = `${fields.providerID ?? "?"}/${fields.modelID ?? "?"} (${fields.npm ?? "?"})`
      const entry = source.markers.get(key) ?? { calls: 0, supports: fields.supportsCacheMarkers === "true" }
      entry.calls += 1
      source.markers.set(key, entry)
      continue
    }

    if (message === "working-set cap applied") {
      const maxBytes = num(fields.maxBytes)
      source.caps.count += 1
      if (fields.scope) source.caps.scopes.add(fields.scope)
      if (maxBytes !== undefined) {
        source.caps.minCap = Math.min(source.caps.minCap, maxBytes)
        source.caps.maxCap = Math.max(source.caps.maxCap, maxBytes)
      }
    }
  }
  return source
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  return sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))]
}

function summarize(source: Source) {
  const ctx = source.usage.map((u) => u.ctx).sort((a, b) => a - b)
  const sum = (pick: (u: Usage) => number) => source.usage.reduce((acc, u) => acc + pick(u), 0)
  const input = sum((u) => u.input)
  const cacheRead = sum((u) => u.cacheRead)
  const cacheWrite = sum((u) => u.cacheWrite)
  const processed = input + cacheRead + cacheWrite
  const bigUncached = source.usage.filter((u) => u.input > BIG_UNCACHED).length
  const dropped = source.usage.filter((u) => u.sdKDroppedCache).length
  return {
    calls: source.usage.length,
    p50: percentile(ctx, 0.5),
    p90: percentile(ctx, 0.9),
    max: ctx.length ? ctx[ctx.length - 1] : 0,
    input,
    cacheRead,
    processed,
    cacheShare: processed > 0 ? cacheRead / processed : 0,
    bigUncached,
    dropped,
    caps: source.caps,
    markers: source.markers,
  }
}

function fmt(n: number): string {
  return n.toLocaleString("en-US")
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function line(label: string, value: string): string {
  return `  ${label.padEnd(34)} ${value}`
}

function report(source: Source, hooks: number) {
  const s = summarize(source)
  console.log(`\n${hooks}. ${source.name}`)
  console.log(line("model calls (cache.usage)", fmt(s.calls)))
  console.log(line("context tokens / call  p50/p90/max", `${fmtTokens(s.p50)} / ${fmtTokens(s.p90)} / ${fmtTokens(s.max)}`))
  console.log(line("cache-read share of input", `${(s.cacheShare * 100).toFixed(1)}%`))
  console.log(line("full-price input (unbilled)", fmtTokens(s.input)))
  console.log(line(`calls re-reading >${fmtTokens(BIG_UNCACHED)} uncached`, fmt(s.bigUncached)))
  if (s.dropped > 0) console.log(line("SDK dropped flat cached_tokens", `${fmt(s.dropped)} calls (would read 0%)`))
  if (s.caps.count > 0) {
    console.log(line("working-set caps applied", `${fmt(s.caps.count)} across ${s.caps.scopes.size} slice(s)`))
    console.log(line("  smallest / largest cap", `${fmtTokens(s.caps.minCap)} / ${fmtTokens(s.caps.maxCap)}`))
  } else {
    console.log(line("working-set caps applied", "0"))
  }
  if (source.sizes) {
    const total = source.sizes.systemChars + source.sizes.toolsChars
    console.log(line("fixed overhead (system+tools)", `${fmtTokens(total)} chars (~${fmtTokens(Math.round(total / 4))} tok)`))
    console.log(line("  system prompt", `${fmtTokens(source.sizes.systemChars)} chars`))
    console.log(
      line("  tool schemas", `${fmtTokens(source.sizes.toolsChars)} chars across ${fmt(source.sizes.toolCount)} tools`),
    )
  }
  if (s.markers.size > 0) {
    console.log("  models seen (cache markers):")
    for (const [model, info] of s.markers) console.log(`    ${info.supports ? "yes" : "no "}  x${fmt(info.calls)}  ${model}`)
  }
}

function compare(summaries: ReturnType<typeof summarize>[], names: string[]) {
  const rows: Array<[string, (s: ReturnType<typeof summarize>) => string]> = [
    ["model calls", (s) => fmt(s.calls)],
    ["context/call p50", (s) => fmtTokens(s.p50)],
    ["context/call p90", (s) => fmtTokens(s.p90)],
    ["cache-read share", (s) => `${(s.cacheShare * 100).toFixed(1)}%`],
    ["full-price input", (s) => fmtTokens(s.input)],
    [">50k uncached calls", (s) => fmt(s.bigUncached)],
  ]
  const before = summaries[0]
  const after = summaries[summaries.length - 1]
  console.log("\nBEFORE -> AFTER")
  for (const [label, pick] of rows) {
    const b = pick(before)
    const a = pick(after)
    console.log(`  ${label.padEnd(24)} ${b.padStart(10)}  ->  ${a.padStart(10)}`)
  }
  console.log(`  (before: ${names[0]}  after: ${names[names.length - 1]})`)
}

const files = resolveSources(process.argv.slice(2))
if (files.length === 0) {
  console.error("No log sources found. Run `mimo --log-level DEBUG` first, or pass a path.")
  process.exit(1)
}

const sources = files.map(readSource).filter((s) => s.usage.length > 0 || s.markers.size > 0 || s.caps.count > 0)
if (sources.length === 0) {
  console.error(`Read ${files.length} log file(s) but found no cache instrumentation. Enable with: mimo --log-level DEBUG`)
  process.exit(1)
}

console.log(`MiMoCode prompt-efficiency report — ${sources.length} source(s) with instrumentation`)
sources.forEach((source, index) => report(source, index + 1))
if (sources.length > 1) compare(sources.map(summarize), sources.map((s) => s.name))
