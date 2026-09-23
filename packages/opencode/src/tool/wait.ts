import { Effect } from "effect"
import z from "zod"
import { stat } from "node:fs/promises"
import * as Tool from "./tool"
import DESCRIPTION from "./wait.txt"

/** A wait longer than this is not a wait, it is a decision to come back later. */
const TIMEOUT_DEFAULT_SECONDS = 300
const TIMEOUT_CEILING_SECONDS = 900
/** How often a condition is re-checked. An artefact that appears does not need to
 * be noticed within milliseconds, and a tight loop spends the machine's time to
 * save none of anyone else's. */
const POLL_INTERVAL_MS = 2000

const parameters = z.object({
  until_path: z
    .string()
    .optional()
    .describe(
      "Absolute path to wait for: returns as soon as it exists AND is non-empty. Prefer this — it returns the moment the artefact is ready rather than at the next interval.",
    ),
  seconds: z
    .number()
    .positive()
    .optional()
    .describe("Wait this many seconds, then return. Use only when nothing observable marks the end of the job."),
  timeout_seconds: z
    .number()
    .positive()
    .optional()
    .describe(`Give up after this many seconds (default ${TIMEOUT_DEFAULT_SECONDS}, ceiling ${TIMEOUT_CEILING_SECONDS}).`),
})

/** Whether the condition holds. A directory has no size to check, so existence is
 * the test for it; anything else must also be non-empty, because a job creates its
 * artefact before it finishes writing it. */
async function ready(path: string): Promise<boolean> {
  try {
    const stats = await stat(path)
    return stats.isDirectory() || stats.size > 0
  } catch {
    return false
  }
}

export const WaitTool = Tool.define(
  "wait",
  Effect.gen(function* () {
    return {
      description: DESCRIPTION,
      parameters,
      execute: (args: z.infer<typeof parameters>, ctx: Tool.Context) =>
        Effect.gen(function* () {
          const until = args.until_path
          if (until === undefined && args.seconds === undefined) {
            return {
              title: "wait",
              metadata: { met: false, waitedSeconds: 0 },
              output: [
                "wait needs a condition: pass `until_path` (preferred) or `seconds`.",
                "Waiting indefinitely is not one of the options — a turn that never returns is a hang, not a wait.",
              ].join("\n"),
            }
          }

          const started = Date.now()
          const budgetMs = Math.min(args.timeout_seconds ?? TIMEOUT_DEFAULT_SECONDS, TIMEOUT_CEILING_SECONDS) * 1000
          // A duration is a condition that never becomes true, so both forms run the
          // same loop and only the end differ: the requested span, or the budget.
          const span = until === undefined ? Math.min((args.seconds ?? 0) * 1000, budgetMs) : budgetMs
          const end = started + span
          // A sleep that completes has done what it was asked; only a condition can
          // come out unmet.
          let met = until === undefined
          while (Date.now() < end && !ctx.abort.aborted) {
            if (until !== undefined && (yield* Effect.promise(() => ready(until)))) {
              met = true
              break
            }
            const remaining = end - Date.now()
            if (remaining <= 0) break
            // Effect's own clock, not a timer: the wait is interruptible like every
            // other step, and it does not need a Node/Bun-specific sleep.
            yield* Effect.sleep(Math.min(POLL_INTERVAL_MS, remaining))
          }
          if (ctx.abort.aborted) met = false

          const waitedSeconds = Math.round((Date.now() - started) / 1000)
          const title = ctx.abort.aborted
            ? `wait interrupted after ${waitedSeconds}s`
            : met
              ? `waited ${waitedSeconds}s`
              : `timed out after ${waitedSeconds}s`
          return {
            title,
            metadata: { met, waitedSeconds },
            output: ctx.abort.aborted
              ? `Interrupted after ${waitedSeconds}s. Nothing was changed by this call.`
              : until === undefined
                ? `Waited ${waitedSeconds}s.`
                : met
                  ? `${until} is ready (waited ${waitedSeconds}s).`
                  : `${until} is still not ready after ${waitedSeconds}s. This is a FACT, not an error: check the job's own log for why it is slow, do something else and come back, or wait again with a larger \`timeout_seconds\`.`,
          }
        }),
    }
  }),
)
