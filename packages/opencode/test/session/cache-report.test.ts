import { describe, expect, test } from "bun:test"
import {
  cacheReport,
  renderCacheReport,
  DEFAULT_CACHE_TTL_MS,
  UNEXPLAINED_FLOOR_TOKENS,
  type CacheReportRow,
} from "../../src/session/cache-report"

const row = (over: Partial<CacheReportRow>): CacheReportRow => ({
  sessionID: "s1", time: 1_000, agent: "build", model: "p/m",
  uncached: 100, cached: 1_000, written: 0, cost: 0, ...over,
})

describe("cacheReport", () => {
  test("the first call of a session is new-content, not a miss", () => {
    const r = cacheReport({ rows: [row({})] })
    expect(r.buckets["new-content"].calls).toBe(1)
    expect(r.buckets["cold-resume"].calls).toBe(0)
  })

  test("idle beyond the TTL is cold-resume", () => {
    const r = cacheReport({ rows: [row({ time: 0 }), row({ time: DEFAULT_CACHE_TTL_MS + 1, uncached: 400_000 })] })
    expect(r.buckets["cold-resume"].calls).toBe(1)
    expect(r.buckets["cold-resume"].uncached).toBe(400_000)
  })

  test("an agent change within the TTL is a profile-switch, not cold", () => {
    const r = cacheReport({ rows: [row({ time: 0 }), row({ time: 5_000, agent: "general" })] })
    expect(r.buckets["profile-switch"].calls).toBe(1)
    expect(r.buckets["cold-resume"].calls).toBe(0)
  })

  test("a model change within the TTL is a model-switch", () => {
    const r = cacheReport({ rows: [row({ time: 0 }), row({ time: 5_000, model: "p/other" })] })
    expect(r.buckets["model-switch"].calls).toBe(1)
  })

  test("two sessions do not compare each other's clocks", () => {
    // The premise the goal gate taught: work from an EARLIER session is not evidence
    // about this one. A session boundary must reset the gap, not reuse it.
    const r = cacheReport({ rows: [row({ sessionID: "a", time: 0 }), row({ sessionID: "b", time: 5_000 })] })
    expect(r.buckets["new-content"].calls).toBe(2)
    expect(r.buckets["cold-resume"].calls).toBe(0)
    expect(r.sessions).toBe(2)
  })

  test("the verdict names the dominant avoidable cause", () => {
    const rows = [
      row({ time: 0, uncached: 100 }),
      row({ time: DEFAULT_CACHE_TTL_MS + 1, uncached: 900, agent: "general" }),
    ]
    const r = cacheReport({ rows })
    expect(r.verdict.state).toBe("unhealthy") // 90% avoidable
    expect(r.verdict.reason).toContain("cold-resume")
  })

  test("a healthy session is mostly new content", () => {
    const rows = [row({ time: 0 }), ...Array.from({ length: 20 }, (_, i) => row({ time: (i + 1) * 1_000, uncached: 1_000 }))]
    const r = cacheReport({ rows })
    expect(r.verdict.state).toBe("healthy")
    expect(r.avoidableShare).toBeLessThan(0.15)
  })

  test("offenders list non-new buckets and large same-profile calls, largest first", () => {
    const rows = [
      // cached is deliberately huge on the big rows so the rotation rule cannot
      // re-bucket them: this test is about the offender mechanism (what counts as
      // notable, and the sort), not about bucket semantics.
      row({ time: 0, uncached: 10, cached: 4_000_000 }),
      row({ time: 1_000, uncached: UNEXPLAINED_FLOOR_TOKENS + 5_000, cached: 4_000_000 }), // big new-content
      row({ time: 2_000, uncached: 20_000, cached: 4_000_000, agent: "general" }),         // profile-switch
      row({ time: 2_500, uncached: UNEXPLAINED_FLOOR_TOKENS + 5, cached: 4_000_000, agent: "general" }), // a big call right after a switch is still new-content
      row({ time: 3_000, uncached: 100, cached: 4_000_000, agent: "general" }),            // small new-content
    ]
    const r = cacheReport({ rows })
    expect(r.offenders).toHaveLength(3)
    expect(r.offenders[0]!.bucket).toBe("new-content")
    expect(r.offenders[0]!.uncached).toBe(UNEXPLAINED_FLOOR_TOKENS + 5_000)
    expect(r.offenders[1]!.bucket).toBe("new-content")
    expect(r.offenders[2]!.bucket).toBe("profile-switch")
  })

  test("a same-profile call the provider barely cached is a prefix-rotation, not new content", () => {
    // The measurement that forced this bucket into existence: ~745k-token calls at
    // cached=0 on a stable profile with a short idle — which neither new content nor
    // an idle gap can explain. A report that only bucketed by gap would have called
    // that "new content" and hidden the real cause.
    const r = cacheReport({
      rows: [row({ time: 0, uncached: 1_000, cached: 400_000 }), row({ time: 4_000, uncached: 745_000, cached: 0 })],
    })
    expect(r.buckets["prefix-rotation"].calls).toBe(1)
    expect(r.buckets["new-content"].calls).toBe(1)
    expect(r.buckets["prefix-rotation"].uncached).toBe(745_000)
    expect(r.verdict.state).toBe("unhealthy")
    expect(r.verdict.reason).toContain("prefix-rotation")
  })

  test("a small same-profile call after a tiny prefix is NOT a rotation", () => {
    // The rotation inference must not fire on a call whose whole payload is plausibly
    // new: uncached 300 against cached 100 with no idle gap is a warm session starting
    // out, not an invalidation.
    const r = cacheReport({ rows: [row({ time: 0, uncached: 100, cached: 50 }), row({ time: 4_000, uncached: 300, cached: 100 })] })
    expect(r.buckets["prefix-rotation"].calls).toBe(0)
    expect(r.buckets["new-content"].calls).toBe(2)
  })

  test("cold-resume wins over rotation: the idle gap is an observed fact", () => {
    const r = cacheReport({
      rows: [row({ time: 0, uncached: 1_000, cached: 400_000 }), row({ time: DEFAULT_CACHE_TTL_MS + 1, uncached: 400_000, cached: 0 })],
    })
    expect(r.buckets["cold-resume"].calls).toBe(1)
    expect(r.buckets["prefix-rotation"].calls).toBe(0)
  })

  test("empty input renders without dividing by zero", () => {
    const r = cacheReport({ rows: [] })
    expect(r.calls).toBe(0)
    expect(r.verdict.state).toBe("healthy")
    expect(renderCacheReport(r)).toContain("verdict: HEALTHY")
  })
})
