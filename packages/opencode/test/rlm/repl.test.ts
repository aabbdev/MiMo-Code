import { describe, expect, test } from "bun:test"
import { injectPayload } from "../../src/rlm/payload"
import { charge, GROUNDING_FLOOR_PCT, groundingOf, lastToolResult, newSpend } from "../../src/tool/repl"

describe("repl sub-call budget", () => {
  test("both bounds refuse, and neither implies the other", () => {
    const base = newSpend(3, 1000)
    // A volume bound a call count alone would not catch.
    expect(() => charge(base, 1, 1001)).toThrow(/volume budget/)
    // A count bound a volume bound alone would not catch: 4 tiny calls.
    expect(() => charge(base, 4, 1)).toThrow(/sub-call budget/)
    expect(charge(base, 1, 400)).toMatchObject({ subcalls: 1, chars: 400 })
  })

  test("a batch that cannot be afforded fails before it half-runs", () => {
    const spent = charge(newSpend(5, 10_000), 4, 4000)
    // One call and 6000 characters remain, so a 2-call batch and a 7000-character
    // single call are both impossible — and neither may partially execute.
    expect(() => charge(spent, 2, 1000)).toThrow(/sub-call budget/)
    expect(() => charge(spent, 1, 7000)).toThrow(/volume budget/)
    expect(charge(spent, 1, 1000)).toMatchObject({ subcalls: 5, chars: 5000 })
  })

  test("the refusal tells the model what to do, not merely that it failed", () => {
    expect(() => charge(newSpend(1, 100), 2, 10)).toThrow(/load the payload again to reset/)
    expect(() => charge(newSpend(5, 100), 1, 200)).toThrow(/load the payload again to reset/)
  })

  test("a VOLUME-only charge leaves the call allowance intact", () => {
    // The screening fix rests on this: batches are charged by volume so the harness
    // stops eating the trajectory's own question allowance.
    const spent = charge(newSpend(2, 1000), 0, 400)
    expect(spent.subcalls).toBe(0)
    expect(spent.chars).toBe(400)
    // …and the volume bound still bites, so it is not a way around the budget.
    expect(() => charge(spent, 0, 700)).toThrow(/volume budget/)
  })

  test("a fresh load resets both counters", () => {
    expect(newSpend(7, 500)).toMatchObject({ subcalls: 0, chars: 0, maxSubcalls: 7, maxSubcallChars: 500 })
  })
})

describe("injectPayload", () => {
  test("exposes the parts, and NOT a second copy of the payload", () => {
    const seen: Record<string, unknown> = {}
    injectPayload(
      { set: (name: string, value: unknown) => void (seen[name] = value) },
      { text: "WHOLE", type: "2-file directory", files: 2, partNames: ["a.cpp", "b.cpp"], partTexts: ["A", "B"] },
    )
    // `context` is a lazy getter over the parts in the kernel prelude. Injecting it
    // as well meant the guest held a 4.5 MB payload twice: +92 MB of process RSS.
    expect(Object.keys(seen).sort()).toEqual(["__joined", "context_part_names", "context_parts"])
    expect(seen.context_parts).toEqual(["A", "B"])
    expect(seen.context_part_names).toEqual(["a.cpp", "b.cpp"])
    expect(seen.context).toBeUndefined()
  })
})

describe("repl grounding", () => {
  // The defect this instrument exists to catch, measured before it moved here: a
  // run that printed names and sizes, made ZERO sub-calls, had seen under 1 % of the
  // payload, and returned a confident 10 500-character description of 119 files
  // built from their filenames — 1 of 5 spot-checks right.
  const loaded = () => ({ observedChars: 0, warned: false })

  test("loading is not an ungrounded answer, so nothing is said before a code step", () => {
    expect(groundingOf(loaded(), 4_000_000, 0, 0).warning).toBeUndefined()
  })

  test("a code step that printed under the floor with no sub-call is warned, once", () => {
    const shown = { observedChars: 3000, warned: false }
    const first = groundingOf(shown, 4_906_634, 0, 1)
    expect(first.warning).toBeDefined()
    expect(first.observedPct).toBeLessThan(GROUNDING_FLOOR_PCT)
    // It names the obligation rather than reporting an alarm.
    expect(first.warning).toContain("never its meaning")
    expect(first.warning).toContain("llm_query")
    // Once: a paragraph repeated every step stops being read.
    expect(groundingOf({ ...shown, warned: true }, 4_906_634, 0, 2).warning).toBeUndefined()
  })

  test("one sub-call clears the condition entirely, whatever was printed", () => {
    expect(groundingOf({ observedChars: 100, warned: false }, 4_906_634, 1, 3).warning).toBeUndefined()
  })

  test("crossing the floor clears it, so a real read is never nagged", () => {
    const read = { observedChars: 300_000, warned: false }
    expect(groundingOf(read, 4_906_634, 0, 4).observedPct).toBeGreaterThan(GROUNDING_FLOOR_PCT)
    expect(groundingOf(read, 4_906_634, 0, 4).warning).toBeUndefined()
  })

  test("an empty payload cannot be ungrounded", () => {
    const out = groundingOf(loaded(), 0, 0, 1)
    expect(out.observedPct).toBe(100)
    expect(out.warning).toBeUndefined()
  })

  test("the share is of the payload, and rides on every step", () => {
    const half = groundingOf({ observedChars: 2_453_317, warned: true }, 4_906_634, 0, 9)
    expect(half.observedPct).toBeCloseTo(50, 6)
  })
})

describe("repl keep: which tool result moves into the kernel", () => {
  const tool = (name: string, output: string, status = "completed") => ({
    type: "tool",
    tool: name,
    callID: `call_${name}`,
    state: { status, output },
  })
  const user = { info: { role: "user" }, parts: [{ type: "text", text: "hi" }] }
  const assistant = (...parts: any[]) => ({ info: { role: "assistant" }, parts })

  test("takes the most recent completed result, not the first", () => {
    const found = lastToolResult([
      assistant(tool("grep", "old hits")),
      assistant(tool("exec", "new hits")),
    ])
    expect(found).toEqual({ text: "new hits", tool: "exec", complete: true })
  })

  test("an unfinished result is not a result", () => {
    // A tool still running has no output to keep, and one that failed has an error.
    expect(lastToolResult([assistant(tool("grep", "partial", "running"))])).toBeUndefined()
    expect(lastToolResult([assistant(tool("grep", "", "completed"))])).toBeUndefined()
  })

  test("its own output is skipped, so keep cannot feed on itself", () => {
    const found = lastToolResult([assistant(tool("grep", "hits"), tool("repl", "REPL OUTPUT"))])
    expect(found?.tool).toBe("grep")
  })

  test("a truncated result points at the FULL text, which is the whole point", () => {
    // The harness saves what it cut and names the file. Keeping the inline summary
    // would hand the kernel exactly what the model already had.
    const found = lastToolResult([
      assistant(tool("grep", "12000 hits…\n\nFull output saved to: /tmp/example/full.txt")),
    ])
    expect(found).toEqual({ text: "/tmp/example/full.txt", tool: "grep", complete: false })
  })

  test("user turns carry no tool parts", () => {
    expect(lastToolResult([user])).toBeUndefined()
  })
})
