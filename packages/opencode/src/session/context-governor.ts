/**
 * Pure predicates for the context-governor decision — the facts every rebuild path
 * must agree on.
 *
 * This module exists because the SAME predicate was inlined on two call sites with a
 * comment warning that they must not drift, and they HAD drifted in shape: one
 * fetched the boundary unconditionally, the other only when a checkpoint existed,
 * and one of them had already shipped a defect (reading a nullable column as
 * `!== undefined`, which was true for every session with a file on disk). A predicate
 * that lives in two places is one bug away from being true in one and false in the
 * other, and the callers cannot tell.
 *
 * Deliberately Effect-free: these take the OBSERVED facts as arguments. The callers
 * own the reads (hasCheckpoint, lastBoundary) — which also means the predicate is
 * testable without a database.
 *
 * It is a TYPE GUARD, not a boolean, and that is not cosmetic: the caller needs the
 * `boundary` value narrowed to a message id afterwards (it feeds
 * `insertRebuildBoundary`), and a plain boolean would force an unsafe cast right
 * after the guard — recreating exactly the unchecked-cast defect this module exists
 * to prevent.
 */

/**
 * Whether an on-disk checkpoint can actually rebuild the context. Both facts are
 * required, and BOTH are checked with truthiness: `lastBoundary` reads a nullable
 * column and has historically returned JS `null` for an unset watermark while its
 * declared type said `MessageID | undefined`, so `boundary !== undefined` was true
 * for EVERY session with a file on disk and the guard degenerated into the bare
 * hasCheckpoint check it was written to replace. Truthiness is the contract, not a
 * style choice.
 *
 * The two false cases mean different things downstream and callers rely on the
 * difference: a missing checkpoint file means there is nothing to rebuild from and
 * the caller MAY degrade to compaction; a present file with no boundary means a
 * template was scaffolded before the writer finished — a NORMAL arrival state — and
 * the caller must not compact on it, it must start-or-wait for the writer.
 */
export function isRebuildUsable(
  hasCheckpoint: boolean,
  boundary: string | null | undefined,
): boundary is string {
  return hasCheckpoint && !!boundary
}
/**
 * Whether a fold pays for itself.
 *
 * The formula is billion-context's (acp-kernel `computeFoldEconomics`), because they
 * measured it against months of production folds and it answers the question our
 * harness has never asked: the checkpoint writer costs 12.6 % of a measured
 * three-day bill (627 calls at 217K tokens each) and nothing ever checked whether a
 * write is repaid by the tokens it stops us carrying.
 *
 *   oneTimeCostUnits = (w - r) * T + q * sigma - r * S
 *   perTurnSavingUnits = (S - sigma) * r
 *   breakevenTurns    = oneTimeCost / perTurnSaving
 *   paidBack          = cadence >= breakevenTurns
 *
 * where S is the tokens the fold removed from the carried context, sigma the tokens
 * the summary itself costs on every later request, T the one-time re-pay (the cold
 * re-read while the prefix restarts), and cadence the turns between folds.
 *
 * ONE weight is OURS, not theirs, and it matters: `r` is the cached/uncached price
 * ratio. Their default 0.1 assumes a 10x discount; the measured ratio on this
 * machine's provider is $0.006/M cached against $0.30/M uncached = 0.02 — a 50x
 * discount. A larger discount means a fold saves LESS per turn and its one-time
 * re-pay weighs MORE, so breakeven is LONGER here than their default would report.
 * Using their 0.1 would understate the payback period by 5x on the saving term.
 * `q` stays a policy weight (how much worse a summary is than the original).
 */
export interface FoldEconomicsInput {
  /** Tokens the fold removed from the carried context (S). */
  folded: number
  /** Tokens the summary itself costs on every later request (sigma). */
  summary: number
  /** One-time re-pay: tokens charged at the cold rate while the prefix restarts (T). */
  repay: number
  /** Turns between folds (k). Omitted when the cadence is not yet observed. */
  cadence?: number
  weights?: { w?: number; r?: number; q?: number }
}

export interface FoldEconomics {
  oneTimeCostUnits: number
  perTurnSavingUnits: number
  breakevenTurns: number
  /** True when the observed cadence already covers the break-even. Undefined when
   *  no cadence was supplied — "unknown", never a silent false. */
  paidBack: boolean | undefined
}

export const FOLD_WEIGHTS = { w: 1, r: 0.02, q: 4 } as const

export function foldEconomics(input: FoldEconomicsInput): FoldEconomics {
  const { w = FOLD_WEIGHTS.w, r = FOLD_WEIGHTS.r, q = FOLD_WEIGHTS.q } = input.weights ?? {}
  const oneTimeCostUnits = (w - r) * input.repay + q * input.summary - r * input.folded
  const perTurnSavingUnits = (input.folded - input.summary) * r
  const breakevenTurns =
    perTurnSavingUnits > 0 ? oneTimeCostUnits / perTurnSavingUnits : Number.POSITIVE_INFINITY
  return {
    oneTimeCostUnits,
    perTurnSavingUnits,
    breakevenTurns,
    paidBack: input.cadence === undefined ? undefined : input.cadence >= breakevenTurns,
  }
}
