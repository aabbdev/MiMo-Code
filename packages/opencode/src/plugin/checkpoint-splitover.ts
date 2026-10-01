import type { Hooks, PluginInput } from "@mimo-ai/plugin"
import { Log } from "../util"
import type { ProjectID } from "../project/schema"
import type { SessionID } from "../session/schema"
import { runValidatorsForCkpt, buildExtractionReflection, buildReflectionMessage } from "../session/checkpoint-retry"
import { checkpointPath, memoryPath } from "../session/checkpoint-paths"
import * as CheckpointContext from "../session/checkpoint-context"

const log = Log.create({ service: "plugin.checkpoint-splitover" })

const EMPTY_CTX: CheckpointContext.CheckpointContext = {
  priorTitles: new Set<string>(),
  expectedRevisions: [],
}

export async function CheckpointSplitoverPlugin(pluginInput: PluginInput): Promise<Hooks> {
  const projectID = pluginInput.project.id as ProjectID

  return {
    "actor.preStop": {
      matcher: { agentType: { include: ["checkpoint-writer"] } },
      run: async (input, output) => {
        // Bookkeeping (CheckpointContext) and file paths (checkpointPath) are keyed
        // on the PARENT session — the writer runs in a child session post-Axis-A but
        // writes to the parent's artifacts. Fall back to input.sessionID for spawn
        // shapes where parent === sessionID (dream/distill don't get the child redirect).
        const sessionID = (input.parentSessionID ?? input.sessionID) as SessionID
        const ctx = CheckpointContext.get(sessionID, input.actorID) ?? EMPTY_CTX
        try {
          const violations = await runValidatorsForCkpt(sessionID, {
            priorTitles: ctx.priorTitles,
            expectedRevisions: ctx.expectedRevisions,
            projectID,
          })
          if (violations.length === 0) return

          const extractRequired = violations.filter((v) => v.severity === "extract-required")
          if (extractRequired.length > 0) {
            // Log the ask. Without this line the only trace of a repair that never
            // converged was the cap warning in actor/spawn.ts, which cannot say WHY
            // the loop was still going — 21 abandoned repairs produced one unhelpful
            // WARN each and nothing a user could act on.
            log.warn("checkpoint over budget; asking the writer to extract", {
              actorID: input.actorID,
              targets: extractRequired.map((v) => `${v.file}: ${v.detail}`),
            })
            output.continue = true
            output.reason = buildExtractionReflection(extractRequired)
            return
          }

          const errors = violations.filter((v) => v.severity === "error")
          if (errors.length > 0) {
            log.warn("checkpoint failed validation; asking the writer to repair", {
              actorID: input.actorID,
              errors: errors.map((v) => `${v.file}: ${v.rule} — ${v.detail}`),
            })
            output.continue = true
            output.reason = buildReflectionMessage(errors, {
              checkpoint: checkpointPath(sessionID),
              memory: memoryPath(projectID),
            })
            return
          }
          // warn-only → fall through, leave output untouched
        } catch (err) {
          log.error("checkpoint-splitover hook failed", { err, sessionID, actorID: input.actorID })
        }
      },
    },
  }
}
