/**
 * Real model spend a tool incurred OUTSIDE the request that called it.
 *
 * `repl` talks to a model on its own, so the session never sees that
 * spend in any request's usage — which means every readout built on message cost
 * silently omitted it. Measured before this existed: a session that spent $0.6601
 * was reported as $0.0151, a 43× understatement, and it was the most expensive
 * session in the database.
 *
 * A tool reports it in its own result metadata and `SessionProcessor` adds it to
 * the calling message's `cost`, so the whole-session aggregate, its route and the
 * sidebar's deltas all pick it up without knowing this exists.
 *
 * Entries are keyed by (provider, model) rather than summed into one figure: a
 * tool may pay two models in one call — a sub-call runs on the lite tier while
 * anything the caller pays for runs on the session's model — and attributing both
 * to one of them would make a per-model view wrong.
 *
 * Same rule as the max-mode ensemble overhead in the processor: added to `cost`
 * and to the per-model metrics, NEVER to `tokens`, which must stay the request's
 * real context footprint or overflow estimation drifts.
 */
export type OverheadEntry = {
  cost: number
  tokensIn: number
  tokensOut: number
  /** Cached input tokens, so a ratio computed from this stays honest. */
  cacheRead: number
  provider: string
  model: string
}

export const OVERHEAD_KEY = "overhead"

/** Merge entries naming the same model — a tool's two models can be the same
 * whenever no lite tier is configured — so a tool can build its list naively. */
export function overheadEntries(entries: OverheadEntry[]): OverheadEntry[] {
  const merged = entries.reduce((acc, entry) => {
    const key = `${entry.provider}/${entry.model}`
    const current = acc.get(key)
    return acc.set(
      key,
      current
        ? {
            ...entry,
            cost: current.cost + entry.cost,
            tokensIn: current.tokensIn + entry.tokensIn,
            tokensOut: current.tokensOut + entry.tokensOut,
            cacheRead: current.cacheRead + entry.cacheRead,
          }
        : entry,
    )
  }, new Map<string, OverheadEntry>())
  return [...merged.values()]
}

function isEntry(value: unknown): value is OverheadEntry {
  if (typeof value !== "object" || value === null) return false
  const entry = value as Record<string, unknown>
  return (
    typeof entry.cost === "number" &&
    typeof entry.tokensIn === "number" &&
    typeof entry.tokensOut === "number" &&
    typeof entry.cacheRead === "number" &&
    typeof entry.provider === "string" &&
    typeof entry.model === "string"
  )
}

/**
 * The entries a tool published, tolerating anything else.
 *
 * Metadata crosses a JSON boundary and comes from a tool that already ran, so a
 * malformed or absent entry means "nothing to add" — never a failed tool call,
 * and never a phantom charge. Empty entries are dropped here, in the one place
 * that decides what counts, so producers can stay naive.
 */
export function overheadOf(metadata: Record<string, any> | undefined): OverheadEntry[] {
  const raw = metadata?.[OVERHEAD_KEY]
  if (!Array.isArray(raw)) return []
  return raw
    .filter(isEntry)
    .filter((entry) => entry.cost > 0 || entry.tokensIn > 0 || entry.tokensOut > 0)
}
