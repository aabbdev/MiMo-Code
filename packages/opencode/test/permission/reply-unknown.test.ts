import { afterEach, describe, expect, test } from "bun:test"
import { Effect, Fiber, Layer } from "effect"
import { Bus } from "../../src/bus"
import * as CrossSpawnSpawner from "../../src/effect/cross-spawn-spawner"
import { Permission } from "../../src/permission"
import { Instance } from "../../src/project/instance"
import { tmpdir } from "../fixture/fixture"
import { Log } from "../../src/util"

void Log.init({ print: false })

afterEach(async () => {
  await Instance.disposeAll()
})

const env = Layer.mergeAll(Permission.layer.pipe(Layer.provide(Bus.layer)), Bus.layer, CrossSpawnSpawner.defaultLayer)

// A request that genuinely needs an ask: empty ruleset, no default rule matches.
function buildRequest() {
  return {
    permission: "read" as never,
    patterns: ["/some/never-allowed-path"],
    always: ["*"],
    metadata: {},
    sessionID: "ses_reply_unknown" as never,
    ruleset: [],
    tool: { messageID: "msg_reply_unknown" as never, callID: "call_reply_unknown" },
  }
}

describe("Permission.reply", () => {
  // The instance that held a request can be disposed and rebuilt underneath the
  // UI, so the reply finds nothing. Reporting that lets the client drop a stale
  // prompt instead of leaving it on screen forever.
  test("reports false for an unknown request instead of silently succeeding", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () =>
        Effect.gen(function* () {
          const perm = yield* Permission.Service
          const ok = yield* perm.reply({ requestID: "per_does_not_exist" as never, reply: "once" })
          expect(ok).toBe(false)
        }).pipe(Effect.provide(env), Effect.runPromise),
    })
  })

  test("reports true and resolves a genuinely pending ask", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () =>
        Effect.gen(function* () {
          const perm = yield* Permission.Service
          const fiber = yield* perm.ask(buildRequest()).pipe(Effect.forkScoped)

          const request = yield* Effect.gen(function* () {
            for (let i = 0; i < 100; i++) {
              const pending = yield* perm.list()
              if (pending.length > 0) return pending[0]
              yield* Effect.sleep("10 millis")
            }
            return undefined
          })
          expect(request).toBeDefined()

          const ok = yield* perm.reply({ requestID: request!.id, reply: "once" })
          expect(ok).toBe(true)
          yield* Fiber.join(fiber)
        }).pipe(Effect.provide(env), Effect.scoped, Effect.runPromise),
    })
  }, 10_000)
})
