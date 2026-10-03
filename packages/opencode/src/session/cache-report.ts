import { Locale, Token } from "../util"

/**
 * Decompose a session's cache behaviour into buckets that NAME the cause of every
 * uncached token.
 *
 * Why this exists: the harness's bill is dominated by tokens re-sent at the UNCACHED
 * rate (measured on this machine: 47 % of a three-day bill, at $0.30/M against
 * $0.006/M cached — 50x). The cache hit RATE says nothing about it (96.5-99.2 %
 * every day); what matters is WHERE the uncached tokens come from, because each
 * cause has a different remedy:
 *
 *   cold-resume      — the provider cache expired on inactivity. Remedy: rebuild
 *                      from the checkpoint on resume (shipped), or accept the re-read.
 *   profile-switch   — the agent changed, so the prefix is a different cache key.
 *                      Remedy: fewer profile alternations, or a cheaper prefix.
 *   model-switch     — same, for the model dimension alone.
 *   prefix-rotation  — same profile, short idle, and the provider served almost
 *                      NOTHING from cache (uncached > 2x cached on a call large
 *                      enough that new content cannot explain it). The prefix was
 *                      invalidated upstream: a tools-hash rotate, provider eviction,
 *                      or a wire change. This bucket exists because the measurement
 *                      that found it showed ~745k-token calls at cached=0 on a
 *                      stable profile — which neither new content nor an idle gap
 *                      can explain, and which a report that only buckets by gap
 *                      would have mislabeled as "new content".
 *   new-content      — genuinely new material (tool results, the user's message). The
 *                      irreducible part; bounded by the working-set governor.
 *
 * Everything here is provider-agnostic: it reports TOKENS and lets the caller price
 * them. The one rate it does apply is the cached/uncached SPLIT that the
 * trajectory DB already records per call (`tokens.input` EXCLUDES the cached part;
 * the total is input + cache.read + cache.write).
 *
 * The thresholds are POLICY, stated here so they are visible and changeable — they
 * are not measured constants, and this file must not pretend otherwise. The measured
 * baseline they were set against: on one real session the cold-resume bucket alone
 * was 55 % of the uncached total, which is what "unhealthy" exists to catch.
 */

export const DEFAULT_CACHE_TTL_MS = 300_000
/** Uncached tokens above which a same-profile call is listed as an offender. */
export const UNEXPLAINED_FLOOR_TOKENS = 50_000
/** Verdict bands over the avoidable share of the uncached total. */
export const HEALTHY_AVOIDABLE_SHARE = 0.15
export const DEGRADED_AVOIDABLE_SHARE = 0.4

export type BucketName = "new-content" | "cold-resume" | "profile-switch" | "model-switch" | "prefix-rotation"

export interface CacheReportRow {
  sessionID: string
  /** Epoch ms of the call. */
  time: number
  agent: string
  /** `providerID/modelID` — the dimension a switch invalidates. */
  model: string
  /** Tokens sent at the UNCACHED rate. */
  uncached: number
  /** Tokens served from the provider cache. */
  cached: number
  /** Tokens written to the provider cache, when the provider reports them. */
  written: number
  cost: number
}

export interface BucketLine {
  bucket: BucketName
  calls: number
  /** Uncached tokens charged to this bucket. */
  uncached: number
  /** Cached tokens these calls also re-read (they are not free). */
  cached: number
  share: number
}

export interface OffenderLine {
  sessionID: string
  time: number
  agent: string
  model: string
  bucket: BucketName
  uncached: number
  /** Idle ms before the call, when the bucket is time-driven. */
  gap: number | undefined
}

export interface CacheReport {
  calls: number
  sessions: number
  /**
   * Per-session wire-churn census, when the caller supplied one. `redundant` is the
   * number of profile keys that map to an already-seen (system, tools) pair — keys
   * that rotated WITHOUT changing a byte on the wire, so every one of them is a
   * prefix the harness rebuilt for nothing.
   */
  churn: Array<{ sessionID: string; wirePairs: number; profiles: number; redundant: number }>
  span: { from: number; to: number } | undefined
  totalUncached: number
  totalCached: number
  totalWritten: number
  buckets: Record<BucketName, BucketLine>
  /** The share of uncached tokens with a named, remediable cause. */
  avoidableShare: number
  verdict: { state: "healthy" | "degraded" | "unhealthy"; reason: string }
  /** The largest uncached calls, largest first, capped by the caller. */
  offenders: OffenderLine[]
}

export interface CacheReportInput {
  /** Rows for ONE OR MORE sessions, already filtered by the caller's window. */
  rows: CacheReportRow[]
  /** Cache TTL in ms; calls idle longer than this are bucketed cold-resume. */
  ttl?: number
  /**
   * Optional wire-churn census per session, from the prefix-snapshot table:
   * how many DISTINCT (system, tools) byte-pairs a session produced, against how
   * many profile keys. The difference is churn on the harness's OWN key without a
   * byte change — measured at 43 of 69 profiles on one real session — which is the
   * part of a prefix-rotation the harness can fix without touching the provider.
   */
  churn?: Record<string, { wirePairs: number; profiles: number }>
}

export function cacheReport(input: CacheReportInput): CacheReport {
  const ttl = input.ttl ?? DEFAULT_CACHE_TTL_MS

  // Rows MUST be grouped per session before any gap is computed. Sorting by time
  // alone interleaves concurrent sessions, and with several sessions active the
  // row before a call is usually ANOTHER session's — which made nearly every call
  // look like a session start and buried every rotation under "new content"
  // (measured: 5,916 of 6,310 calls bucketed new-content, 94%). A gap is only
  // meaningful between two calls of the SAME session.
  const bySession = new Map<string, CacheReportRow[]>()
  let first = Infinity
  let last = -Infinity
  for (const row of input.rows) {
    const list = bySession.get(row.sessionID)
    if (list) list.push(row)
    else bySession.set(row.sessionID, [row])
    if (row.time < first) first = row.time
    if (row.time > last) last = row.time
  }

  const buckets: Record<BucketName, BucketLine> = {
    "new-content": { bucket: "new-content", calls: 0, uncached: 0, cached: 0, share: 0 },
    "cold-resume": { bucket: "cold-resume", calls: 0, uncached: 0, cached: 0, share: 0 },
    "profile-switch": { bucket: "profile-switch", calls: 0, uncached: 0, cached: 0, share: 0 },
    "model-switch": { bucket: "model-switch", calls: 0, uncached: 0, cached: 0, share: 0 },
    "prefix-rotation": { bucket: "prefix-rotation", calls: 0, uncached: 0, cached: 0, share: 0 },
  }
  const offenders: OffenderLine[] = []

  let totalUncached = 0
  let totalCached = 0
  let totalWritten = 0

  for (const list of bySession.values()) {
    list.sort((a, b) => a.time - b.time)
    let previous: CacheReportRow | undefined
    for (const row of list) {
      totalUncached += row.uncached
      totalCached += row.cached
      totalWritten += row.written

      const firstOfSession = previous === undefined
      const gap = firstOfSession ? undefined : row.time - previous!.time
      // Precedence is the certainty of the cause, not the size of the cost: an idle gap
      // and a profile/model change are OBSERVED facts, so they win. A same-profile call
      // the provider barely cached is an INFERENCE (something rotated), so it is checked
      // last — a small call after a tiny prefix must not be flagged as a rotation.
      const rotation =
        !firstOfSession &&
        row.uncached > 2 * row.cached &&
        row.uncached >= UNEXPLAINED_FLOOR_TOKENS
      const bucket = firstOfSession
        ? "new-content"
        : gap! > ttl
          ? "cold-resume"
          : previous!.agent !== row.agent
            ? "profile-switch"
            : previous!.model !== row.model
              ? "model-switch"
              : rotation
                ? "prefix-rotation"
                : "new-content"

      const line = buckets[bucket]
      line.calls += 1
      line.uncached += row.uncached
      line.cached += row.cached

      const notable = bucket !== "new-content" || row.uncached >= UNEXPLAINED_FLOOR_TOKENS
      if (notable)
        offenders.push({
          sessionID: row.sessionID,
          time: row.time,
          agent: row.agent,
          model: row.model,
          bucket,
          uncached: row.uncached,
          gap,
        })

      previous = row
    }
  }

  for (const line of Object.values(buckets)) line.share = share(line.uncached, totalUncached)

  const avoidable =
    buckets["cold-resume"].uncached +
    buckets["profile-switch"].uncached +
    buckets["model-switch"].uncached +
    buckets["prefix-rotation"].uncached
  const avoidableShare = share(avoidable, totalUncached)
  const state =
    avoidableShare > DEGRADED_AVOIDABLE_SHARE
      ? "unhealthy"
      : avoidableShare > HEALTHY_AVOIDABLE_SHARE
        ? "degraded"
        : "healthy"
  const dominant = (["cold-resume", "profile-switch", "model-switch", "prefix-rotation"] as const)
    .map((name) => buckets[name])
    .sort((a, b) => b.uncached - a.uncached)[0]!
  const reason =
    state === "healthy"
      ? `avoidable share ${pct(avoidableShare)} of uncached tokens; the rest is genuinely new content`
      : `${pct(avoidableShare)} of uncached tokens are avoidable, dominated by ${dominant.bucket} (${pct(dominant.share)})`

  offenders.sort((a, b) => b.uncached - a.uncached)

  const churn = Object.entries(input.churn ?? {})
    .map(([sessionID, c]) => ({
      sessionID,
      wirePairs: c.wirePairs,
      profiles: c.profiles,
      redundant: c.profiles - c.wirePairs,
    }))
    .sort((a, b) => b.redundant - a.redundant)

  return {
    calls: input.rows.length,
    sessions: bySession.size,
    churn,
    span: input.rows.length > 0 ? { from: first, to: last } : undefined,
    totalUncached,
    totalCached,
    totalWritten,
    buckets,
    avoidableShare,
    verdict: { state, reason },
    offenders,
  }
}

/** Render the report as the compact text the CLI prints. */
export function renderCacheReport(report: CacheReport, options: { limit?: number } = {}): string {
  const lines: string[] = []
  const span = report.span ? new Date(report.span.from).toISOString() : "-"
  lines.push(`calls=${report.calls} sessions=${report.sessions} from=${span}`)
  lines.push(
    `uncached=${Token.format(report.totalUncached)} cached=${Token.format(report.totalCached)} written=${Token.format(report.totalWritten)}`,
  )
  lines.push("")
  lines.push("bucket           calls  uncached  share")
  for (const line of Object.values(report.buckets).sort((a, b) => b.uncached - a.uncached)) {
    lines.push(
      line.bucket.padEnd(16) + String(line.calls).padStart(6) + Token.format(line.uncached).padStart(10) + pct(line.share).padStart(7),
    )
  }
  if (report.churn.some((c) => c.redundant > 0)) {
    lines.push("")
    lines.push("wire churn (profile keys vs distinct system+tools bytes):")
    for (const c of report.churn.slice(0, 5))
      lines.push(
        `  ${c.sessionID.slice(0, 22)} profiles=${String(c.profiles).padStart(4)} wirePairs=${String(c.wirePairs).padStart(4)} redundant=${String(c.redundant).padStart(4)}`,
      )
  }
  lines.push("")
  lines.push(`verdict: ${report.verdict.state.toUpperCase()} — ${report.verdict.reason}`)
  const limit = options.limit ?? 10
  const shown = report.offenders.slice(0, limit)
  if (shown.length > 0) {
    lines.push("")
    lines.push(`largest uncached calls (top ${shown.length}):`)
    for (const o of shown)
      lines.push(
        `  ${new Date(o.time).toISOString()} ${o.bucket.padEnd(14)} ${Token.format(o.uncached).padStart(8)} ${o.agent} ${o.model}` +
          (o.gap !== undefined ? ` idle=${Locale.duration(o.gap)}` : ""),
      )
  }
  return lines.join("\n")
}

function share(part: number, whole: number) {
  return whole > 0 ? part / whole : 0
}

function pct(value: number) {
  return `${Math.round(value * 100)}%`
}