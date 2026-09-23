import { describe, expect, test } from "bun:test"
import { injectPayload } from "../../src/rlm/payload"
import { charge, newSpend } from "../../src/tool/repl"

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

  test("a fresh load resets both counters", () => {
    expect(newSpend(7, 500)).toMatchObject({ subcalls: 0, chars: 0, maxSubcalls: 7, maxSubcallChars: 500 })
  })
})

describe("injectPayload", () => {
  test("exposes exactly the three globals the guidance teaches", () => {
    const seen: Record<string, unknown> = {}
    injectPayload(
      { set: (name: string, value: unknown) => void (seen[name] = value) },
      { text: "WHOLE", type: "2-file directory", files: 2, partNames: ["a.cpp", "b.cpp"], partTexts: ["A", "B"] },
    )
    expect(Object.keys(seen).sort()).toEqual(["context", "context_part_names", "context_parts"])
    expect(seen.context).toBe("WHOLE")
    expect(seen.context_parts).toEqual(["A", "B"])
    expect(seen.context_part_names).toEqual(["a.cpp", "b.cpp"])
  })
})
