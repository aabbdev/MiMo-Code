import { describe, test, expect } from "bun:test"
import { workingSetCap, WORKING_SET_MIN_CAP_BYTES } from "../../src/tool/working-set"

const BASE = 50 * 1024
const BUDGET = 100 * 1024

describe("workingSetCap", () => {
  test("budget <= 0 disables the governor (legacy per-result cap)", () => {
    expect(workingSetCap({ usedBytes: 0, baseCap: BASE, budget: 0 })).toBe(BASE)
    expect(workingSetCap({ usedBytes: 10_000_000, baseCap: BASE, budget: 0 })).toBe(BASE)
  })

  test("returns the full base cap while there is room", () => {
    expect(workingSetCap({ usedBytes: 0, baseCap: BASE, budget: BUDGET })).toBe(BASE)
    expect(workingSetCap({ usedBytes: BUDGET - BASE, baseCap: BASE, budget: BUDGET })).toBe(BASE)
  })

  test("shrinks the cap to exactly what is left once the budget is tight", () => {
    const used = BUDGET - 20 * 1024
    expect(workingSetCap({ usedBytes: used, baseCap: BASE, budget: BUDGET })).toBe(20 * 1024)
  })

  test("floors at minCap once the budget is spent (never zero)", () => {
    expect(workingSetCap({ usedBytes: BUDGET, baseCap: BASE, budget: BUDGET })).toBe(WORKING_SET_MIN_CAP_BYTES)
    expect(workingSetCap({ usedBytes: BUDGET * 10, baseCap: BASE, budget: BUDGET })).toBe(WORKING_SET_MIN_CAP_BYTES)
  })

  test("honours a caller cap below the floor", () => {
    expect(workingSetCap({ usedBytes: BUDGET * 10, baseCap: 1024, budget: BUDGET })).toBe(1024)
  })

  test("never returns a value above baseCap or below minCap", () => {
    for (const used of [0, 1, BUDGET - 1, BUDGET, BUDGET + 1, BUDGET * 100]) {
      const cap = workingSetCap({ usedBytes: used, baseCap: BASE, budget: BUDGET })
      expect(cap).toBeLessThanOrEqual(BASE)
      expect(cap).toBeGreaterThanOrEqual(WORKING_SET_MIN_CAP_BYTES)
      expect(cap).toBeGreaterThan(0)
    }
  })
})
