import { afterEach, describe, expect } from "bun:test"
import { Effect, Layer } from "effect"
import * as CrossSpawnSpawner from "../../src/effect/cross-spawn-spawner"
import { Session } from "../../src/session"
import { MessageID } from "../../src/session/schema"
import { Instance } from "../../src/project/instance"
import { provideTmpdirInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"

afterEach(async () => {
  await Instance.disposeAll()
})

const env = Layer.mergeAll(CrossSpawnSpawner.defaultLayer, Session.defaultLayer)
const it = testEffect(env)

describe("Session.totalCost", () => {
  /**
   * Writes `count` assistant messages costing `each`, walking the same shape the
   * processor writes. Returns the session.
   */
  const seed = (count: number, each: number) =>
    Effect.gen(function* () {
      const ssn = yield* Session.Service
      const info = yield* ssn.create({ title: "Cost" })
      let t = Date.now()
      for (let i = 0; i < count; i++) {
        const parent = yield* ssn.updateMessage({
          id: MessageID.ascending(),
          role: "user" as const,
          sessionID: info.id,
          agent: "build",
          model: { providerID: "test" as never, modelID: "test-model" as never },
          time: { created: t++ },
        })
        yield* ssn.updateMessage({
          id: MessageID.ascending(),
          role: "assistant" as const,
          sessionID: info.id,
          agentID: "main",
          agent: "build",
          mode: "primary",
          modelID: "test-model" as never,
          providerID: "test" as never,
          parentID: parent.id,
          time: { created: t++, completed: t++ },
          finish: "end_turn",
          tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
          path: { cwd: "/", root: "/" },
          cost: each,
        })
      }
      return info
    })

  it.live("sums every assistant message in the session", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const ssn = yield* Session.Service
        const info = yield* seed(3, 0.5)
        expect(yield* ssn.totalCost({ sessionID: info.id })).toBeCloseTo(1.5, 6)
      }),
    ),
  )

  // The whole point: a total summed from a page of messages under-reports. The
  // TUI held the newest 100 and showed $0.50 for a session that had spent
  // $17.30; this asserts the count is not page-bounded.
  it.live("counts messages far older than any page", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const ssn = yield* Session.Service
        const info = yield* seed(120, 0.01)
        // 120 assistant messages = 240 rows, well past the 100-message page the
        // TUI hydrates and past the 1000-row server default too in spirit.
        expect(yield* ssn.totalCost({ sessionID: info.id })).toBeCloseTo(1.2, 6)
        const page = yield* ssn.messages({ sessionID: info.id, limit: 100 })
        const fromPage = page.reduce((sum, m) => sum + (m.info.role === "assistant" ? (m.info.cost ?? 0) : 0), 0)
        expect(fromPage).toBeLessThan(1.2)
      }),
    ),
  )

  it.live("ignores user messages and counts other sessions separately", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const ssn = yield* Session.Service
        const a = yield* seed(2, 0.25)
        const b = yield* seed(1, 4)
        expect(yield* ssn.totalCost({ sessionID: a.id })).toBeCloseTo(0.5, 6)
        expect(yield* ssn.totalCost({ sessionID: b.id })).toBeCloseTo(4, 6)
      }),
    ),
  )
})
