/**
 * Working-set budget for inline tool output.
 *
 * Within a cache epoch the transcript is append-only, so the only place to bound
 * how much tool output accumulates into EVERY request is at INSERTION — the
 * moment a result is stored. Mutating old results afterwards (prune, rebuild)
 * invalidates the provider prompt-cache prefix from the first touched message,
 * which is precisely the full-price re-read this governor exists to avoid.
 *
 * Measured on real sessions, ~33 results at the 50 KiB per-result cap
 * (`preview.MAX_BYTES`) accumulate a ~1.6 MiB working set — the bulk of a
 * 300-900k-token context. Capping the AGGREGATE, not just each result, is what
 * bounds it.
 *
 * `workingSetCap` returns the byte cap for the NEXT result given how much inline
 * tool output the working set already holds: the full base cap while there is
 * room, a shrinking cap as the budget fills, and a small stub cap once spent.
 * The full text is STILL spilled to disk and reachable via Read, so nothing is
 * lost — only its inline footprint shrinks. Pure and dependency-free.
 */

/**
 * Floor for a single result once the aggregate budget is spent: enough for a
 * meaningful head+tail preview plus the "full output saved to <path>" hint, so
 * the model can still choose to Read the file. Never zero — a zero cap yields
 * empty content, which some providers reject as a non-empty-content violation.
 */
export const WORKING_SET_MIN_CAP_BYTES = 4 * 1024

/**
 * Per-result cap for the next tool output.
 *
 * - `budget <= 0` → disabled, return `baseCap` (legacy behaviour).
 * - `baseCap <= minCap` → the caller asked for less than the floor; honour it.
 * - room for the full result → `baseCap`.
 * - room for only part of it → exactly what is left (never below `minCap`).
 * - nothing left → `minCap` (a stub; the result is spilled, not dropped).
 */
export function workingSetCap(input: {
  usedBytes: number
  baseCap: number
  budget: number
  minCap?: number
}): number {
  const minCap = input.minCap ?? WORKING_SET_MIN_CAP_BYTES
  if (input.budget <= 0) return input.baseCap
  if (input.baseCap <= minCap) return input.baseCap
  const remaining = input.budget - input.usedBytes
  if (remaining >= input.baseCap) return input.baseCap
  if (remaining <= minCap) return minCap
  return remaining
}
