import fs from "fs/promises"
import path from "path"

/**
 * The fold ledger — the data foundation of the block model.
 *
 * A rebuild folds a range of the transcript into a summary and keeps a tail. Today
 * that fold is INVISIBLE after the fact: the checkpoint is rewritten to cover the
 * whole session, so nothing records how many folds happened, what each one cost,
 * or what range each covered. The block model (billion-context's CompressionBlock,
 * reduced to what this harness needs) starts by making folds ADDRESSABLE:
 *
 *   - one ledger entry per fold, in order, so generations are countable;
 *   - the covered RANGE as (firstMessageID, lastMessageID) — the transcript is
 *     ordered, so a range is two ids and `history around` addresses the original
 *     text between them;
 *   - the measured economics per fold (folded, summary, break-even turns), which is
 *     the data the cadence-side gate needs before it can ask whether a WRITE pays
 *     back.
 *
 * Deliberately NOT here yet: the multi-fold VIEW at assembly (rewriting the message
 * list per request), tiering, and block GC. Those change the cache-prefix machinery
 * and are separate increments. This file is pure data + IO, no Effect.
 */

export interface ContextBlock {
  blockId: string
  /** The covered range, inclusive — `history around` restores its originals. */
  firstMessageID: string
  lastMessageID: string
  /** Tokens the fold removed from the carried context (S). */
  foldedTokens: number
  /** Tokens the summary itself costs on every later request (sigma). */
  summaryTokens: number
  /** Break-even turns from foldEconomics at fold time. */
  breakevenTurns: number
  /** 1-based: the Nth fold of this session. Never reset. */
  generation: number
  createdAt: number
}

export interface FoldLedger {
  version: 1
  blocks: ContextBlock[]
}

export const FOLD_LEDGER_VERSION = 1

export function emptyLedger(): FoldLedger {
  return { version: FOLD_LEDGER_VERSION, blocks: [] }
}

/**
 * Append a fold. `generation` is derived — the Nth entry — so a caller cannot
 * desynchronise it, and a ledger rebuilt from disk keeps numbering stable.
 */
export function appendFold(
  ledger: FoldLedger,
  fold: Omit<ContextBlock, "blockId" | "generation" | "createdAt">,
  now = Date.now(),
): FoldLedger {
  const block: ContextBlock = {
    ...fold,
    blockId: `f${String(ledger.blocks.length + 1).padStart(3, "0")}`,
    generation: ledger.blocks.length + 1,
    createdAt: now,
  }
  return { version: FOLD_LEDGER_VERSION, blocks: [...ledger.blocks, block] }
}

export function foldLedgerPath(metaDir: string) {
  return path.join(metaDir, "fold-ledger.json")
}

export async function loadFoldLedger(metaDir: string): Promise<FoldLedger> {
  const text = await fs.readFile(foldLedgerPath(metaDir), "utf-8").catch(() => "")
  if (!text.trim()) return emptyLedger()
  try {
    const parsed = JSON.parse(text) as FoldLedger
    if (parsed?.version !== FOLD_LEDGER_VERSION || !Array.isArray(parsed.blocks)) return emptyLedger()
    return parsed
  } catch {
    // A corrupt ledger must not block a fold: start empty, the next append rewrites it.
    return emptyLedger()
  }
}

export async function saveFoldLedger(metaDir: string, ledger: FoldLedger): Promise<void> {
  await fs.mkdir(metaDir, { recursive: true })
  const target = foldLedgerPath(metaDir)
  // Rename-then-create: a torn in-place write would be indistinguishable from an
  // empty ledger, which appendFold then treats as generation 1 again.
  const tmp = `${target}.tmp-${process.pid}`
  await fs.writeFile(tmp, JSON.stringify(ledger, null, 2))
  await fs.rename(tmp, target)
}

/**
 * The cadence the cadence-side gate needs: turns-per-fold so far, from the ledger
 * plus the calls observed since the last fold. Returns undefined until two folds
 * exist — one fold is not a cadence, it is an event.
 */
export function observedCadenceTurns(
  ledger: FoldLedger,
  callsSinceLastFold: number,
): number | undefined {
  const folds = ledger.blocks.length
  if (folds < 2) return undefined
  return Math.round(callsSinceLastFold / (folds - 1))
}