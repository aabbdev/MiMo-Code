import { describe, expect, test } from "bun:test"
import { makeScreener } from "../../src/rlm/host"

const index = (count: number) => Array.from({ length: count }, (_, i) => ({ name: `part${i}`, preview: `preview ${i}` }))

describe("makeScreener", () => {
  test("shows previews, names the question, and returns the indices the model picked", async () => {
    const prompts: string[] = []
    const screen = makeScreener(async (messages) => {
      prompts.push(String(messages[0]!.content))
      return "[1] [3]"
    })
    expect(await screen("what is the contract?", index(4))).toEqual([1, 3])
    expect(prompts[0]).toContain("Question: what is the contract?")
    expect(prompts[0]).toContain("[1] part1")
    // Previews, not content: the whole point is that the first pass is small.
    expect(prompts[0]).toContain("preview 1")
  })

  test("out-of-range and repeated numbers are dropped", async () => {
    const screen = makeScreener(async () => "[0] [99] [0] [-1]")
    expect(await screen("q", index(2))).toEqual([0])
  })

  test("NONE selects nothing rather than everything", async () => {
    const screen = makeScreener(async () => "NONE")
    expect(await screen("q", index(5))).toEqual([])
  })

  test("a large payload is screened in batches, so no single call is unbounded", async () => {
    let calls = 0
    const screen = makeScreener(async () => {
      calls += 1
      return "NONE"
    })
    await screen("q", index(130))
    expect(calls).toBe(3) // 60 + 60 + 10
  })

  test("screening spends from the same allowance as any other model call", async () => {
    const charged: number[] = []
    const screen = makeScreener(
      async () => "NONE",
      (chars) => charged.push(chars),
    )
    await screen("q", index(3))
    expect(charged).toHaveLength(1)
    expect(charged[0]!).toBeGreaterThan(100)
  })

  test("an exhausted budget refuses the screen instead of running it free", async () => {
    const screen = makeScreener(
      async () => "[0]",
      () => {
        throw new Error("sub-call budget exhausted")
      },
    )
    await expect(screen("q", index(2))).rejects.toThrow(/budget exhausted/)
  })

  test("a malformed index is treated as empty rather than crashing the guest", async () => {
    const screen = makeScreener(async () => "[0]")
    expect(await screen("q", undefined)).toEqual([])
    expect(await screen("q", "not an array")).toEqual([])
  })
})
