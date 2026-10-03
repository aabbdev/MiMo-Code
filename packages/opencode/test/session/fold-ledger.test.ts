import fs from "fs/promises"
import { describe, expect, test } from "bun:test"
import {
  appendFold,
  emptyLedger,
  foldLedgerPath,
  loadFoldLedger,
  observedCadenceTurns,
  saveFoldLedger,
} from "../../src/session/fold-ledger"
import { tmpdir } from "../fixture/fixture"

const fold = (over: Partial<Parameters<typeof appendFold>[1]> = {}) => ({
  firstMessageID: "msg_1",
  lastMessageID: "msg_9",
  foldedTokens: 745_000,
  summaryTokens: 31_000,
  breakevenTurns: 10,
  ...over,
})

describe("fold ledger — pure model", () => {
  test("appendFold derives the blockId and the generation — the caller cannot desynchronise them", () => {
    let ledger = emptyLedger()
    ledger = appendFold(ledger, fold())
    ledger = appendFold(ledger, fold({ firstMessageID: "msg_10", lastMessageID: "msg_20" }))
    expect(ledger.blocks.map((b) => [b.blockId, b.generation])).toEqual([
      ["f001", 1],
      ["f002", 2],
    ])
  })

  test("one fold is an event, not a cadence", () => {
    // The cadence-side gate needs this distinction: undefined means "unknown",
    // never a silently small number that makes an expensive fold look paid back.
    expect(observedCadenceTurns(appendFold(emptyLedger(), fold()), 40)).toBeUndefined()
  })

  test("the cadence divides the observed calls by the folds-1 interval", () => {
    let ledger = emptyLedger()
    for (let i = 0; i < 3; i++) ledger = appendFold(ledger, fold())
    expect(observedCadenceTurns(ledger, 40)).toBe(20)
  })
})

describe("fold ledger — IO", () => {
  test("save then load round-trips with generations intact", async () => {
    await using tmp = await tmpdir()
    let ledger = emptyLedger()
    for (let i = 0; i < 2; i++)
      ledger = appendFold(ledger, fold({ firstMessageID: `msg_${i}`, lastMessageID: `msg_${i + 5}` }))
    await saveFoldLedger(tmp.path, ledger)
    const loaded = await loadFoldLedger(tmp.path)
    expect(loaded.version).toBe(1)
    expect(loaded.blocks).toEqual(ledger.blocks)
    expect(foldLedgerPath(tmp.path)).toContain("fold-ledger.json")
  })

  test("a corrupt ledger starts empty instead of blocking a fold", async () => {
    await using tmp = await tmpdir()
    await fs.writeFile(foldLedgerPath(tmp.path), "{not json at all")
    const loaded = await loadFoldLedger(tmp.path)
    expect(loaded.blocks).toEqual([])
    expect(loaded.version).toBe(1)
    // And appending after the corruption heals the file: the next save rewrites it.
    await saveFoldLedger(tmp.path, appendFold(loaded, fold()))
    const healed = await loadFoldLedger(tmp.path)
    expect(healed.blocks).toHaveLength(1)
  })

  test("an absent ledger is empty, not an error", async () => {
    await using tmp = await tmpdir()
    const loaded = await loadFoldLedger(tmp.path)
    expect(loaded.blocks).toEqual([])
  })
})