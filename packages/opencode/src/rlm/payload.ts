/**
 * Turning files into a payload, for every consumer of the kernel.
 *
 * Kept out of the tools because both layers need it and they must agree exactly:
 * the automated `rlm` consumer and the session-native `repl`. If they diverged,
 * "the same payload" would mean two different things depending on which one
 * loaded it, and every cost and coverage number would be comparing unlike things.
 *
 * The rules here are also where the safety lives: a directory walk is handed to
 * sub-models — a third party — so an IMPLICIT collection must be conservative,
 * while an explicitly requested path is the agent's own decision and carries the
 * same trust its `read` tool already has.
 */
import os from "os"
import path from "path"
import fs from "node:fs/promises"
import { Instance } from "../project/instance"

const MAX_FILE_BYTES = 10 * 1024 * 1024
const MAX_DIR_BYTES = 16 * 1024 * 1024
const MAX_DIR_FILES = 1000
/** Ceiling on a merged multi-root payload. Each root is bounded on its own, but
 * the kernel holds `context` and `context_parts` as separate strings, so N roots
 * multiply: this keeps the realm inside its 256 MiB heap. */
const MAX_PAYLOAD_BYTES = 24 * 1024 * 1024
/** A part never exceeds this, so a sub-call about one part always fits. */
const CHUNK_CHARS = 48_000

const SKIP_DIRS = new Set(["node_modules", "dist", "build", ".venv", "__pycache__", ".cache", ".mimocode"])
const SKIP_EXT = new Set([
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz",
  ".wasm", ".so", ".o", ".a", ".bin", ".exe", ".dll", ".dylib", ".class", ".jar",
  ".mp3", ".mp4", ".mov", ".wav", ".woff", ".woff2", ".ttf", ".otf", ".eot",
])
const SKIP_NAMES = new Set(["credentials.json", "id_rsa", "id_ed25519", "id_ecdsa", "netrc", "secrets.json"])
const SKIP_SECRET_EXT = new Set([".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk", ".asc"])

export type Payload = {
  text: string
  type: string
  files: number
  partNames: string[]
  partTexts: string[]
}

/** Worktree-relative or absolute, but it must land inside the worktree or the OS
 * temp dir — the same jail `exec`'s raw file primitives use, so a payload cannot
 * be talked out of the project tree. */
export async function resolveInside(input: string): Promise<string> {
  const roots = [Instance.directory, os.tmpdir()]
  const abs = path.resolve(Instance.directory, input)
  const realRoots = await Promise.all(roots.map((root) => fs.realpath(root).catch(() => root)))
  const real = await fs.realpath(abs).catch(() => undefined)
  if (!real) throw new Error(`no such file or directory: ${JSON.stringify(input)}`)
  const inside = realRoots.some((root) => real === root || real.startsWith(root + path.sep))
  if (!inside) throw new Error(`path is outside the worktree: ${JSON.stringify(input)}`)
  return real
}

/**
 * Split at line boundaries. A part that cuts a statement in half makes a sub-LLM
 * answer about half a function, and the seam is invisible in the answer.
 */
export function chunkText(name: string, body: string): Array<{ name: string; text: string }> {
  if (body.length <= CHUNK_CHARS) return [{ name, text: body }]
  const out: Array<{ name: string; text: string }> = []
  let current = ""
  for (const line of body.split("\n")) {
    if (current.length > 0 && current.length + line.length + 1 > CHUNK_CHARS) {
      out.push({ name: `${name}#${out.length + 1}`, text: current })
      current = ""
    }
    current += line + "\n"
  }
  if (current.length > 0) out.push({ name: `${name}#${out.length + 1}`, text: current })
  return out
}

export async function readDirectory(root: string): Promise<{
  text: string
  files: number
  names: string[]
  parts: string[]
}> {
  const entries: Array<{ name: string; text: string }> = []
  let bytes = 0
  let files = 0
  const walk = async (dir: string) => {
    if (bytes >= MAX_DIR_BYTES || files >= MAX_DIR_FILES) return
    const listing = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
    for (const entry of listing) {
      if (bytes >= MAX_DIR_BYTES || files >= MAX_DIR_FILES) return
      // Any dot-entry is skipped, file or directory: dotfiles are where
      // credentials live, and a directory payload is handed to sub-models.
      if (entry.name.startsWith(".")) continue
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        // `build`, `build-cpp`, `build-diag` … are all generated trees, and a
        // payload that pulls them in answers questions about compiled output.
        if (SKIP_DIRS.has(entry.name) || /^build[-_]/.test(entry.name)) continue
        await walk(full)
        continue
      }
      if (!entry.isFile()) continue
      if (SKIP_NAMES.has(entry.name)) continue
      const extension = path.extname(entry.name).toLowerCase()
      if (SKIP_EXT.has(extension) || SKIP_SECRET_EXT.has(extension)) continue
      const stat = await fs.stat(full).catch(() => undefined)
      if (!stat || stat.size > MAX_FILE_BYTES) continue
      const body = await fs.readFile(full, "utf8").catch(() => undefined)
      if (body === undefined) continue
      const remaining = MAX_DIR_BYTES - bytes
      const bounded = body.length > remaining ? body.slice(0, remaining) : body
      entries.push(...chunkText(path.relative(root, full), bounded))
      bytes += bounded.length
      files += 1
    }
  }
  const stat = await fs.stat(root)
  if (stat.isFile()) {
    const body = await fs.readFile(root, "utf8")
    if (body.length > MAX_FILE_BYTES) throw new Error(`file is ${body.length} characters, over the ${MAX_FILE_BYTES} limit`)
    entries.push(...chunkText(path.basename(root), body))
    return { text: body, files: 1, names: entries.map((e) => e.name), parts: entries.map((e) => e.text) }
  }
  await walk(root)
  if (entries.length === 0) throw new Error(`no readable text files under ${JSON.stringify(root)}`)
  return {
    text: entries.map((e) => `### ${e.name}\n${e.text}`).join("\n\n"),
    files,
    names: entries.map((e) => e.name),
    parts: entries.map((e) => e.text),
  }
}

/** Load one or more roots into a single payload. With several roots the part
 * names carry the tree, or two `lib/…` names become indistinguishable and an
 * answer can attribute a file to the wrong tree. */
export async function loadPayload(input: { path?: string; paths?: string[]; text?: string }): Promise<Payload> {
  if (input.text !== undefined) {
    const chunks = chunkText("text", input.text)
    return {
      text: input.text,
      type: "text block",
      files: 1,
      partNames: chunks.map((c) => c.name),
      partTexts: chunks.map((c) => c.text),
    }
  }
  const roots = input.paths ?? (input.path !== undefined ? [input.path] : [])
  if (roots.length === 0) throw new Error("provide `path`, `paths` or `text`")
  const read = await Promise.all(
    roots.map(async (given) => {
      const resolved = await resolveInside(given)
      const out = await readDirectory(resolved)
      const stat = await fs.stat(resolved)
      const prefix = roots.length > 1 ? `${given.replace(/[/\\]+$/, "")}/` : ""
      return {
        names: out.names.map((name) => prefix + name),
        parts: out.parts,
        files: out.files,
        kind: stat.isDirectory() ? `${out.files}-file directory` : "file",
      }
    }),
  )
  const partNames = read.flatMap((entry) => entry.names)
  const partTexts = read.flatMap((entry) => entry.parts)
  const text = partNames.map((name, index) => `### ${name}\n${partTexts[index]}`).join("\n\n")
  if (text.length > MAX_PAYLOAD_BYTES) {
    throw new Error(`payload is ${text.length} characters, over the ${MAX_PAYLOAD_BYTES} limit — narrow the paths`)
  }
  const files = read.reduce((sum, entry) => sum + entry.files, 0)
  return {
    text,
    type: roots.length > 1 ? `${roots.length} trees, ${files} files` : read[0]!.kind,
    files,
    partNames,
    partTexts,
  }
}

/** Inject a loaded payload into a realm as the three globals the prompt teaches. */
export function injectPayload(kernel: { set: (name: string, value: unknown) => void }, payload: Payload): void {
  kernel.set("context", payload.text)
  // The unit of work, exposed as data rather than described in prose: a measured
  // run enumerated every file correctly but had nothing to fan out over, so it
  // answered from filenames. A part is a file, or a line-aligned slice of a large
  // one, and a sub-call takes a batch of them.
  kernel.set("context_parts", payload.partTexts)
  kernel.set("context_part_names", payload.partNames)
}
