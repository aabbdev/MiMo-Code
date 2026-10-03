import { describe, expect, test } from "bun:test"
import { foldEconomics, isRebuildUsable, resolveTaskSessionID } from "../../src/session/context-governor"
import { isDelegatedWorker } from "../../src/agent/config"

describe("isRebuildUsable", () => {
  test("both facts are required", () => {
    expect(isRebuildUsable(true, "msg_1")).toBe(true)
    expect(isRebuildUsable(false, "msg_1")).toBe(false)
    expect(isRebuildUsable(true, undefined)).toBe(false)
    expect(isRebuildUsable(false, undefined)).toBe(false)
  })

  test("a JS null boundary is NOT usable — the historical defect", () => {
    // `lastBoundary` reads a nullable column and returned JS null for an unset
    // watermark while its declared type said `MessageID | undefined`. Under
    // `boundary !== undefined` this case was treated as usable for EVERY session
    // with a file on disk, degenerating the guard into the bare hasCheckpoint
    // check. Truthiness is the contract, so null is false here BY TEST.
    expect(isRebuildUsable(true, null)).toBe(false)
  })

  test("an empty boundary is not usable", () => {
    expect(isRebuildUsable(true, "")).toBe(false)
  })

  test("narrows the boundary, so the caller needs no cast afterwards", () => {
    // The guard exists partly for this: the value feeds insertRebuildBoundary as a
    // message id, and a boolean predicate would push an unsafe cast right after the
    // check — the same unchecked-cast defect that produced the null case above.
    const boundary: string | undefined = "msg_9"
    if (isRebuildUsable(true, boundary)) {
      const id: string = boundary
      expect(id).toBe("msg_9")
    } else {
      throw new Error("usable input must narrow")
    }
  })
})
describe("foldEconomics", () => {
  const FOLD = 100_000, SUMMARY = 10_000, REPAY = 10_000
  // w=1, r=0.02, q=4: oneTime = 0.98*10k + 4*10k - 0.02*100k = 9800+40000-2000 = 47800
  //                 perTurn = 90k*0.02 = 1800 -> breakeven = 26.6 turns
  const econ = () => foldEconomics({ folded: FOLD, summary: SUMMARY, repay: REPAY })

  test("the arithmetic matches the formula with OUR measured weights", () => {
    expect(econ().oneTimeCostUnits).toBeCloseTo(47_800)
    expect(econ().perTurnSavingUnits).toBeCloseTo(1_800)
    expect(econ().breakevenTurns).toBeCloseTo(47_800 / 1_800, 3)
  })

  test("the measured r=0.02 stretches the payback vs bili's r=0.1 default", () => {
    // Their 0.1 assumes a 10x discount; the measured ratio here is 50x. A fold must
    // therefore survive LONGER to pay back on this provider, and a tool that reported
    // the borrowed default would understate the payback period 5x on the saving term.
    const ours = foldEconomics({ folded: FOLD, summary: SUMMARY, repay: REPAY })
    const theirs = foldEconomics({ folded: FOLD, summary: SUMMARY, repay: REPAY, weights: { w: 1, r: 0.1, q: 4 } })
    expect(ours.breakevenTurns).toBeGreaterThan(theirs.breakevenTurns)
    // The saving term scales by r alone, so the understatement is more than 5x.
    expect(ours.breakevenTurns / theirs.breakevenTurns).toBeGreaterThan(5)
  })

  test("paidBack is true at the boundary and false just below", () => {
    const turns = econ().breakevenTurns
    expect(foldEconomics({ folded: FOLD, summary: SUMMARY, repay: REPAY, cadence: turns }).paidBack).toBe(true)
    expect(foldEconomics({ folded: FOLD, summary: SUMMARY, repay: REPAY, cadence: turns - 1 }).paidBack).toBe(false)
  })

  test("an unobserved cadence is UNKNOWN, not false", () => {
    // A silent false would read as "the fold does not pay back" and invite deleting
    // the checkpoint path on the basis of missing data.
    expect(foldEconomics({ folded: FOLD, summary: SUMMARY, repay: REPAY }).paidBack).toBeUndefined()
  })

  test("a fold that removes nothing can never pay back", () => {
    const r = foldEconomics({ folded: 5_000, summary: 10_000, repay: 1_000, cadence: 1_000 })
    expect(r.breakevenTurns).toBe(Number.POSITIVE_INFINITY)
    expect(r.paidBack).toBe(false)
  })
})

describe("isDelegatedWorker", () => {
  test("keys on the agent's declared mode, not the spawn shape", () => {
    expect(isDelegatedWorker({ name: "general", mode: "subagent" })).toBe(true)
    // The case that inverts under peer spawning: a subagent registered with
    // mode:"peer" is STILL a delegated worker, because its agent says so.
    expect(isDelegatedWorker({ name: "general", mode: "subagent" })).toBe(true)
    expect(isDelegatedWorker({ name: "build", mode: "primary" })).toBe(false)
    expect(isDelegatedWorker({ name: "build", mode: "peer" as never })).toBe(false)
  })

  test("an unresolvable agent fails OPEN", () => {
    expect(isDelegatedWorker(undefined)).toBe(false)
  })
})

describe("resolveTaskSessionID", () => {
  const PARENT = "ses_parent", CHILD = "ses_child"
  const general = { name: "general", mode: "subagent" as const }
  const writer = { name: "checkpoint-writer", mode: "subagent" as const }
  const build = { name: "build", mode: "primary" as const }
  const child = { parentID: PARENT }

  test("an explicit session_id always wins", () => {
    expect(resolveTaskSessionID({ explicit: "ses_x", sessionID: CHILD, agent: general, self: child })).toBe("ses_x")
  })

  test("a delegated worker in its own session addresses the OWNING tree", () => {
    expect(resolveTaskSessionID({ sessionID: CHILD, agent: general, self: child })).toBe(PARENT)
  })

  test("a shared-session subagent keeps today's behaviour (no parent to redirect to)", () => {
    // Today a subagent shares the parent session, so ctx.sessionID IS the tree; the
    // rule must not move it.
    expect(resolveTaskSessionID({ sessionID: PARENT, agent: general, self: { parentID: null } })).toBe(PARENT)
  })

  test("system-spawned agents keep their own store, even with a parent", () => {
    // The checkpoint writer runs in an Axis-A child deliberately and addresses its
    // own store today; its declared mode is "subagent", so the delegated rule alone
    // would move it. The exclusion is what keeps its behaviour unchanged.
    expect(resolveTaskSessionID({ sessionID: CHILD, agent: writer, self: child })).toBe(CHILD)
  })

  test("a primary agent never redirects", () => {
    expect(resolveTaskSessionID({ sessionID: CHILD, agent: build, self: child })).toBe(CHILD)
  })

  test("an unresolvable agent keeps the running session", () => {
    expect(resolveTaskSessionID({ sessionID: CHILD, self: child })).toBe(CHILD)
  })
})
