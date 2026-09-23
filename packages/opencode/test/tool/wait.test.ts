import { describe, expect } from "bun:test"
import { Effect, Layer } from "effect"
import path from "path"
import fs from "fs/promises"
import { WaitTool } from "../../src/tool/wait"
import { Tool } from "../../src/tool"
import { Truncate } from "../../src/tool"
import { SessionID, MessageID } from "../../src/session/schema"
import { LSP } from "../../src/lsp"
import { AppFileSystem } from "@mimo-ai/shared/filesystem"
import { Bus } from "../../src/bus"
import { Format } from "../../src/format"
import { Agent } from "../../src/agent/agent"
import * as CrossSpawnSpawner from "../../src/effect/cross-spawn-spawner"
import { provideTmpdirInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"

const ctx = {
  sessionID: SessionID.make("ses_test-wait-session"),
  messageID: MessageID.make(""),
  callID: "",
  agent: "build",
  abort: AbortSignal.any([]),
  messages: [],
  metadata: () => Effect.void,
  ask: () => Effect.void,
}

const it = testEffect(
  Layer.mergeAll(
    LSP.defaultLayer,
    AppFileSystem.defaultLayer,
    Bus.layer,
    Format.defaultLayer,
    CrossSpawnSpawner.defaultLayer,
    Truncate.defaultLayer,
    Agent.defaultLayer,
  ),
)

const run = Effect.fn("WaitToolTest.run")(function* (
  args: Tool.InferParameters<typeof WaitTool>,
  next: Tool.Context = ctx,
) {
  const info = yield* WaitTool
  const tool = yield* info.init()
  return yield* tool.execute(args, next)
})

describe("tool.wait", () => {
  // The defect this tool exists to remove, measured on this harness's own history:
  // 3 143 waiting calls forming 1 640 chains — the model re-issuing `sleep 50;
  // check` because `exec_command`'s yield budget TERMINATES a long command rather
  // than yielding, so a job that outlives the budget can only be polled for.

  it.live("a condition already met returns at once, without rounding up to the interval", () =>
    provideTmpdirInstance((dir) =>
      Effect.gen(function* () {
        const ready = path.join(dir, "final-audit.json")
        yield* Effect.promise(() => fs.writeFile(ready, "{}"))
        const started = Date.now()
        const out = yield* run({ until_path: ready })
        expect(out.metadata).toMatchObject({ met: true })
        expect(Date.now() - started).toBeLessThan(1000)
        expect(out.output).toContain("is ready")
      }),
    ),
  )

  it.live("an empty file is NOT ready, because a job creates its artefact before writing it", () =>
    provideTmpdirInstance((dir) =>
      Effect.gen(function* () {
        const partial = path.join(dir, "final-validation.json")
        yield* Effect.promise(() => fs.writeFile(partial, ""))
        const out = yield* run({ until_path: partial, timeout_seconds: 1 })
        expect(out.metadata).toMatchObject({ met: false })
        // Reported as a fact, not as a failure: the model must be able to choose
        // between waiting again and doing something else.
        expect(out.output).toContain("FACT, not an error")
      }),
    ),
  )

  it.live("a bare duration returns met, and is clamped by the timeout", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const started = Date.now()
        const out = yield* run({ seconds: 30, timeout_seconds: 1 })
        // Clamped, not refused: the call takes the wait as far as it can.
        expect(Date.now() - started).toBeLessThan(3000)
        expect(out.metadata).toMatchObject({ met: true })
      }),
    ),
  )

  it.live("no condition is refused in words rather than guessed at", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const out = yield* run({})
        expect(out.metadata).toMatchObject({ met: false })
        expect(out.output).toContain("until_path")
        expect(out.output).toContain("hang, not a wait")
      }),
    ),
  )
})
