import { describe, expect, test } from "bun:test"
import { overheadEntries, overheadOf, type OverheadEntry } from "../../src/tool/overhead"

const entry = (over: Partial<OverheadEntry> = {}): OverheadEntry => ({
  cost: 0.25,
  tokensIn: 1000,
  tokensOut: 100,
  cacheRead: 0,
  provider: "togetherai",
  model: "deepseek-ai/DeepSeek-V4.1-Flash",
  ...over,
})

describe("overheadOf", () => {
  test("a tool that reports nothing adds nothing, whatever its metadata holds", () => {
    expect(overheadOf(undefined)).toEqual([])
    expect(overheadOf({})).toEqual([])
    expect(overheadOf({ costUsd: 0.25 })).toEqual([])
    expect(overheadOf({ overhead: "0.25" })).toEqual([])
    expect(overheadOf({ overhead: { cost: 0.25 } })).toEqual([])
  })

  test("a malformed entry inside a valid list yields nothing rather than throwing", () => {
    // Metadata crosses a JSON boundary and comes from a tool that already ran, so
    // a bad entry has to mean "you get nothing" — a throw here would fail a tool
    // call whose work is already done.
    expect(overheadOf({ overhead: [null, 7, "x", { cost: 1 }, entry()] })).toEqual([entry()])
  })

  test("an entry that spent nothing is dropped, so no phantom call is published", () => {
    expect(overheadOf({ overhead: [entry({ cost: 0, tokensIn: 0, tokensOut: 0 })] })).toEqual([])
    // Still real: zero cost but real tokens is a spend worth reporting once a
    // provider prices it at nothing.
    expect(overheadOf({ overhead: [entry({ cost: 0 })] })).toHaveLength(1)
  })

  test("the cache split survives, because a cost without it is wrong by up to 50x", () => {
    const entries = overheadOf({ overhead: [entry({ cacheRead: 900 })] })
    expect(entries[0]!.cacheRead).toBe(900)
  })
})

describe("overheadEntries", () => {
  test("two spends on the same model merge, because a per-model view must not list it twice", () => {
    const merged = overheadEntries([entry({ cost: 1, tokensIn: 10, cacheRead: 4 }), entry({ cost: 2, tokensIn: 20, cacheRead: 6 })])
    expect(merged).toHaveLength(1)
    expect(merged[0]).toEqual(entry({ cost: 3, tokensIn: 30, tokensOut: 200, cacheRead: 10 }))
  })

  test("two spends on different models stay apart, because one call can pay two models", () => {
    const merged = overheadEntries([entry({ model: "root-model" }), entry({ model: "lite-model" })])
    expect(merged.map((value) => value.model).sort()).toEqual(["lite-model", "root-model"])
  })

  test("the same model served by two providers is two entries", () => {
    const merged = overheadEntries([entry({ provider: "a" }), entry({ provider: "b" })])
    expect(merged).toHaveLength(2)
  })
})
