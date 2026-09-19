import { NodePath } from "@effect/platform-node"
import { Cause, Duration, Effect, Layer, Schedule, Context } from "effect"
import path from "path"
import type { Agent } from "../agent/agent"
import { AppFileSystem } from "@mimo-ai/shared/filesystem"
import { evaluate } from "@/permission/evaluate"
import { Identifier } from "../id/id"
import { Log } from "../util"
import { ToolID } from "./schema"
import { TRUNCATION_DIR } from "./truncation-dir"
import { Flag } from "@/flag/flag"
import { workingSetCap } from "./working-set"
import {
  MAX_BYTES,
  MAX_LINES,
  previewToolOutput,
  type PreviewOptions,
  type PreviewResult,
} from "./preview"

export { MAX_BYTES, MAX_LINES, previewToolOutput }
export type { PreviewResult }
export type Options = PreviewOptions & {
  /**
   * The tool already applied its OWN per-result budget and reported
   * `metadata.truncated`. Its output must still be ACCOUNTED against the
   * session's working-set budget, and still be shrunk once that budget is spent
   * — but it must NOT be re-truncated while the aggregate has room, because that
   * would only add a second omission hint.
   *
   * Without this distinction the governor was inert for every self-truncating
   * tool (`read`, `bash`, `grep` — which set `truncated` on every result), i.e.
   * for exactly the tools whose output accumulates into the working set.
   */
  selfTruncated?: boolean
  /**
   * Text that must close the output, appended when truncation would otherwise
   * drop it. A tool that renders a delimited document (`exec` wraps its result
   * in `<exec>…</exec>`) loses the closing tag to a head-only cut, handing the
   * model an unbalanced envelope. Declaring the closing text lets the producer
   * keep ownership of its own shape while the truncator stays generic.
   *
   * Skipped when the kept text already contains its final line — a head+tail cut
   * can preserve the original close, and appending it twice would be worse.
   */
  closing?: string
}

const log = Log.create({ service: "truncation" })
const RETENTION = Duration.days(7)

export const DIR = TRUNCATION_DIR
export const GLOB = path.join(TRUNCATION_DIR, "*")

export type Result = { content: string; truncated: false } | { content: string; truncated: true; outputPath: string }

function hasActorTool(agent?: Agent.Info) {
  if (!agent?.permission) return false
  return evaluate("actor", "*", agent.permission).action !== "deny"
}

export function formatToolTruncationHint(file: string, outcome: "success" | "error", agent?: Agent.Info): string {
  const result = outcome === "error" ? "failed" : "succeeded"
  return hasActorTool(agent)
    ? `The tool call ${result} but the output was truncated. Full output saved to: ${file}\nUse the actor tool to have explore agent process this file with Grep and Read (with offset/limit). Do NOT read the full file yourself - delegate to save context.`
    : `The tool call ${result} but the output was truncated. Full output saved to: ${file}\nUse Grep to search the full content or Read with offset/limit to view specific sections.`
}

export interface Interface {
  readonly cleanup: () => Effect.Effect<void>
  readonly write: (text: string) => Effect.Effect<string>
  /**
   * Same preview as `previewToolOutput`; when truncated, writes the full text
   * to the truncation directory and appends the tool-result file-path hint.
   *
   * When `sessionID` is supplied, the per-result cap is additionally governed by
   * the working-set budget (see tool/working-set.ts): the aggregate inline tool
   * output cannot grow past `MIMOCODE_WORKING_SET_BUDGET_BYTES`, so a long
   * session cannot accumulate a megabyte of results into every request. The
   * bound is applied here, at insertion — the stored result is never mutated.
   *
   * The budget is accounted per ACTOR SLICE (`sessionID` + `actorID`), not per
   * session: subagents share the parent's sessionID but hold their own message
   * slice, so their tool output never enters the parent's context and must not
   * consume the parent's budget. `actorID` omitted means the main slice.
   */
  readonly output: (
    text: string,
    options?: Options,
    agent?: Agent.Info,
    sessionID?: string,
    actorID?: string,
  ) => Effect.Effect<Result>
  /**
   * Reset every actor slice's working-set accounting for a session. Call on
   * context rebuild (checkpoint discard + rebuild) — the one point where the
   * working set legitimately starts over. Within an epoch a counter only grows.
   */
  readonly reset: (sessionID: string) => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/Truncate") {}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const fs = yield* AppFileSystem.Service

    // Inline tool-output bytes already in each actor slice's working set, keyed
    // `${sessionID}:${actorID ?? "main"}`. Never decremented within an epoch;
    // cleared by `reset` at a rebuild. Per-slice rather than per-session because
    // a subagent shares the parent's sessionID but holds its own message slice.
    const usedByScope = new Map<string, number>()
    const scopeKey = (sessionID: string, actorID?: string) => `${sessionID}:${actorID ?? "main"}`

    const reset = Effect.fn("Truncate.reset")(function* (sessionID: string) {
      const prefix = `${sessionID}:`
      for (const key of usedByScope.keys()) {
        if (key.startsWith(prefix)) usedByScope.delete(key)
      }
    })

    const cleanup = Effect.fn("Truncate.cleanup")(function* () {
      const cutoff = Identifier.timestamp(
        Identifier.create("tool", "ascending", Date.now() - Duration.toMillis(RETENTION)),
      )
      const entries = yield* fs.readDirectory(TRUNCATION_DIR).pipe(
        Effect.map((all) => all.filter((name) => name.startsWith("tool_"))),
        Effect.catch(() => Effect.succeed([])),
      )
      for (const entry of entries) {
        if (Identifier.timestamp(entry) >= cutoff) continue
        yield* fs.remove(path.join(TRUNCATION_DIR, entry)).pipe(Effect.catch(() => Effect.void))
      }
    })

    const write = Effect.fn("Truncate.write")(function* (text: string) {
      const file = path.join(TRUNCATION_DIR, ToolID.ascending())
      yield* fs.ensureDir(TRUNCATION_DIR).pipe(Effect.orDie)
      yield* fs.writeFileString(file, text).pipe(Effect.orDie)
      return file
    })

    const output = Effect.fn("Truncate.output")(function* (
      text: string,
      options: Options = {},
      agent?: Agent.Info,
      sessionID?: string,
      actorID?: string,
    ) {
      const baseCap = options.maxBytes ?? MAX_BYTES
      const budget = Flag.MIMOCODE_WORKING_SET_BUDGET_BYTES
      const scope = sessionID !== undefined && budget > 0 ? scopeKey(sessionID, actorID) : undefined
      const usedBytes = scope !== undefined ? (usedByScope.get(scope) ?? 0) : 0
      const maxBytes = scope !== undefined ? workingSetCap({ usedBytes, baseCap, budget }) : baseCap

      // A self-truncated result is left as-is WHILE the aggregate has room (the
      // tool already applied its own per-result budget); it is only shrunk once
      // the working-set budget itself is spent. It is ALWAYS accounted below.
      const trustTool = options.selfTruncated === true && maxBytes >= baseCap
      if (scope !== undefined && maxBytes < baseCap) {
        // Measurement hook: shows the governor biting as the working set fills.
        log.debug("working-set cap applied", { scope, usedBytes, budget, baseCap, maxBytes })
      }
      const preview = trustTool
        ? ({ content: text, truncated: false } as const)
        : previewToolOutput(text, { ...options, maxBytes })
      if (scope !== undefined) {
        usedByScope.set(scope, usedBytes + Buffer.byteLength(preview.content, "utf-8"))
      }
      if (!preview.truncated) {
        return { content: preview.content, truncated: false } as const
      }
      const file = yield* write(text)
      const hint = formatToolTruncationHint(file, options.outcome ?? "success", agent)
      // Skip when the kept text already carries the closing's final line: a
      // head+tail cut preserves the original close, and doubling it would be
      // worse than the unbalance this option exists to prevent.
      const finalLine = options.closing?.split("\n").at(-1)
      const body =
        finalLine !== undefined && finalLine !== "" && !preview.content.includes(finalLine)
          ? `${preview.content}\n${options.closing}`
          : preview.content
      return {
        content: `${body}\n\n${hint}`,
        truncated: true,
        outputPath: file,
      } as const
    })

    yield* cleanup().pipe(
      Effect.catchCause((cause) => {
        log.error("truncation cleanup failed", { cause: Cause.pretty(cause) })
        return Effect.void
      }),
      Effect.repeat(Schedule.spaced(Duration.hours(1))),
      Effect.delay(Duration.minutes(1)),
      Effect.forkScoped,
    )

    return Service.of({ cleanup, write, output, reset })
  }),
)

export const defaultLayer = layer.pipe(Layer.provide(AppFileSystem.defaultLayer), Layer.provide(NodePath.layer))
