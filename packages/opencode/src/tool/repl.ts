/**
 * `repl` — the session's own kernel. First-class, not a tool that returns an answer.
 *
 * This is the architecture the two papers describe and oh-my-pi ships: the payload
 * is EXTERNALIZED into a persistent environment, and the agent's own loop is the
 * loop that iterates over it. There is no second prompt and no second loop — the
 * tool that owns a completion (`rlm`) still exists, but as an automated CONSUMER
 * of the same kernel, for the one thing a session cannot do: sample several
 * trajectories and select among them.
 *
 * Three operations, and the split between the first two is the whole idea:
 *
 *  - `load` puts a payload into the kernel and returns METADATA ONLY — type,
 *    characters, part count, a short prefix. The payload never enters the prompt.
 *    It also sets the sub-call budget for that payload.
 *  - `code` runs a step and returns a bounded prefix of what it produced. The
 *    bound is not an optimization: it is what forces the payload to be examined
 *    through variables and sub-calls instead of being printed back into the
 *    window.
 *  - `reset` drops the realm, because a first-class kernel means a realm that
 *    lives for the session, and anything that lives that long needs a way out.
 */
import z from "zod"
import { Effect } from "effect"
import type { Tool as AiTool } from "ai"
import { ModelID, ProviderID } from "../provider/schema"
import { Agent } from "../agent/agent"
import { EffectBridge } from "@/effect"
import { acquire, disposeSession } from "../rlm/kernel"
import { makeScreener, makeSubCallerFactory, SCREEN_BATCH_CEILING } from "../rlm/host"
import { overheadEntries, type OverheadEntry } from "./overhead"
import { injectPayload, loadPayload } from "../rlm/payload"
import { TOOL_SCRIPT_ALIASES, TOOL_SCRIPT_EXCLUDED, toolScriptRegistry } from "./tool-script-ref"
import type { HarnessMode } from "./gpt"
import * as Tool from "./tool"
import DESCRIPTION from "./repl.txt"

/** How a step's output is shown back. The framing is load-bearing and was
 * measured: with a bare tag wrapper the model treated the echo as text to
 * complete — it fabricated an echo of its own and copied the trailing reminder
 * back — and it stopped reporting its confidence the moment the history filled
 * with code. An explicit "this is the result of YOUR step" framing scored 2/3 on
 * compliance where the tag scored 0/3. */
const ECHO_CHARS = 3000
const ERROR_ECHO_CHARS = 1200
const PRELUDE_PREFIX_CHARS = 600

const MAX_SUBCALLS_DEFAULT = 40
const MAX_SUBCALLS_CEILING = 200
const SUBCALL_CHARS_DEFAULT = 2_000_000
const SUBCALL_CHARS_CEILING = 20_000_000
const SUBCALL_CALL_CHARS = 400_000

/** Nested tool calls allowed per step. Same order of magnitude as `exec`'s, and
 * per step rather than per kernel: a session kernel can run hundreds of steps, so
 * a kernel-wide allowance would be no allowance at all. */
const MAX_TOOL_CALLS_DEFAULT = 50
const MAX_TOOL_CALLS_CEILING = 500

/** One payload's sub-call spend. Reset by each `load`, so the budget belongs to
 * the payload rather than to a call — which is what a kernel that outlives a call
 * requires. */
export type Spend = { subcalls: number; chars: number; maxSubcalls: number; maxSubcallChars: number }

export function newSpend(maxSubcalls: number, maxSubcallChars: number): Spend {
  return { subcalls: 0, chars: 0, maxSubcalls, maxSubcallChars }
}

/**
 * Charge a sub-call, or refuse it in words the model can act on.
 *
 * Both bounds are load-bearing and neither implies the other: a COUNT alone is
 * not a cost bound (40 calls of 400 000 characters is ~4M tokens of input), and a
 * volume alone permits a thousand tiny calls. Pure, so the arithmetic is testable
 * without a model.
 */
export function charge(state: Spend, count: number, volume: number): Spend {
  if (state.subcalls + count > state.maxSubcalls) {
    throw new Error(
      `sub-call budget exhausted (${state.subcalls} of ${state.maxSubcalls} used). Stop calling llm_query and work with what you already have, or load the payload again to reset the budget.`,
    )
  }
  if (state.chars + volume > state.maxSubcallChars) {
    throw new Error(
      `sub-call volume budget exhausted (${state.chars} of ${state.maxSubcallChars} characters already sent). Send fewer or shorter prompts, or load the payload again to reset the budget.`,
    )
  }
  return { ...state, subcalls: state.subcalls + count, chars: state.chars + volume }
}

/** Per-session state. The kernel is keyed by session; the budget has to live
 * beside it, and it is deliberately module-level: this becomes an Effect service
 * the day a second consumer needs it, and not before. */
const spends = new Map<string, Spend>()

function renderStep(step: { value?: unknown; logs: string[]; error?: string }): { rendered: string; shown: number } {
  const framing = "This is the result of YOUR previous step, not text to continue. Do not repeat it."
  if (step.error) {
    return {
      rendered: `REPL OUTPUT (error, ${step.error.length} chars):\n${step.error.slice(0, ERROR_ECHO_CHARS)}\n\n${framing}`,
      shown: 0,
    }
  }
  const parts: string[] = []
  if (step.value !== undefined) parts.push(`returned: ${typeof step.value === "string" ? step.value : JSON.stringify(step.value)}`)
  for (const line of step.logs) parts.push(`log: ${line}`)
  const body = parts.join("\n")
  const visible = body.slice(0, ECHO_CHARS)
  const cut = body.length - visible.length
  return {
    rendered: `REPL OUTPUT (${body.length} chars, ${parts.length} entries):\n${visible}${
      cut > 0 ? `\n… [${cut} more characters not shown — log a smaller slice, or use llm_query on the variable]` : ""
    }\n\n${framing}`,
    shown: visible.length,
  }
}

/** Stable metadata keys for every outcome — `Tool.define` infers the metadata
 * type from the returned object, so an error return with a different key set
 * makes the success return fail to typecheck. */
type ReplMetadata = {
  error?: string
  reset?: boolean
  payloadChars?: number
  parts?: number
  nestedCalls?: number
  subcallsLeft?: number
  charsLeft?: number
  subModel?: string
  /** What this call's sub-calls cost, in the form `SessionProcessor` adds to the
   * session's own total. Only the sub-calls: in session mode the root turns ARE
   * the agent's own messages, so charging them here would count them twice. */
  overhead?: OverheadEntry[]
}

function replMetadata(over: Partial<ReplMetadata>): ReplMetadata {
  return { ...over }
}

export const ReplTool = Tool.define(
  "repl",
  Effect.gen(function* () {
    const agents = yield* Agent.Service
    // Captured at definition time: execute must have R = never, so it cannot yield
    // Provider/LLM itself.
    const subCaller = yield* makeSubCallerFactory()
    return {
      description: DESCRIPTION,
      parameters: z.object({
        load: z
          .object({
            path: z.string().optional(),
            paths: z.array(z.string()).optional(),
            text: z.string().optional(),
            max_subcalls: z.number().int().min(1).max(MAX_SUBCALLS_CEILING).optional(),
            max_subcall_chars: z.number().int().min(1000).max(SUBCALL_CHARS_CEILING).optional(),
          })
          .optional()
          .describe(
            `Externalize a payload into the kernel and return METADATA ONLY. Re-loading replaces the payload and resets the sub-call budget.`,
          ),
        code: z.string().optional().describe("JavaScript to run in the persistent kernel. Variables survive between calls."),
        max_tool_calls: z
          .number()
          .int()
          .min(1)
          .max(MAX_TOOL_CALLS_CEILING)
          .optional()
          .describe(
            `Nested tool calls allowed in THIS step (default ${MAX_TOOL_CALLS_DEFAULT}). Your own tools are reachable as \`tools.<name>(args)\` — the same permission pipeline as a direct call.`,
          ),
        reset: z.boolean().optional().describe("Drop the kernel and its budget, freeing the payload's memory immediately."),
        model: z.string().optional().describe("Model for the sub-calls. Defaults to the lite/small model, else the session's own."),
      }),
      execute: (
        params: {
          load?: { path?: string; paths?: string[]; text?: string; max_subcalls?: number; max_subcall_chars?: number }
          code?: string
          max_tool_calls?: number
          reset?: boolean
          model?: string
        },
        ctx: Tool.Context,
      ) =>
        Effect.gen(function* () {
          const realmID = `${ctx.sessionID}@repl`
          // Every return below a step that ran carries what the sub-calls cost,
          // including a refusal after it: the provider billed those tokens whether
          // or not the step succeeded, and a spend that vanishes on the failure
          // path is the same defect this accounting exists to remove.
          let sideCost: OverheadEntry[] = []
          const reply = (output: string, over: Partial<ReplMetadata> = {}) => ({
            title: "repl",
            metadata: replMetadata({ ...(sideCost.length > 0 ? { overhead: sideCost } : {}), ...over }),
            output,
          })
          const refuse = (message: string) => reply(`repl could not run: ${message}`, { error: message })

          if (params.reset) {
            disposeSession(realmID)
            spends.delete(ctx.sessionID)
            return reply("kernel dropped; the next `load` starts a fresh one", { reset: true })
          }

          if (!params.load && params.code === undefined) return refuse("provide `load`, `code` or `reset`")

          const users = ctx.messages.flatMap((message) => (message.info.role === "user" ? [message.info] : []))
          const user = users[users.length - 1]
          if (!user) return refuse("no user message in this session")
          const agent = yield* agents.get(ctx.agent)
          const bridge = yield* EffectBridge.make()
          // Shared wiring, not a second copy of it: the sub-model resolution, the
          // replacing system prompt and the no-tools stream live in rlm/host.ts.
          const caller = yield* Effect.promise(() =>
            subCaller.build({
              sessionID: ctx.sessionID,
              agent,
              user,
              actorID: ctx.actorID,
              override: params.model,
            }),
          )
          const { completeSub, subModel } = caller

          let spend = spends.get(ctx.sessionID) ?? newSpend(MAX_SUBCALLS_DEFAULT, SUBCALL_CHARS_DEFAULT)
          const host = {
            llmQuery: async (prompt: unknown) => {
              const text = typeof prompt === "string" ? prompt : JSON.stringify(prompt)
              if (text.length > SUBCALL_CALL_CHARS) {
                throw new Error(`prompt is ${text.length} characters, over the ${SUBCALL_CALL_CHARS}-character limit per sub-call. Split it into smaller chunks.`)
              }
              spend = charge(spend, 1, text.length)
              spends.set(ctx.sessionID, spend)
              return completeSub([{ role: "user", content: text }])
            },
            llmQueryBatched: async (prompts: unknown) => {
              if (!Array.isArray(prompts)) throw new Error("llm_query_batched expects an array of prompt strings")
              const texts = prompts.map((prompt) => (typeof prompt === "string" ? prompt : JSON.stringify(prompt)))
              // Charge the batch up front: a batch that cannot be afforded must fail
              // before it half-runs, or the caller sees a phantom budget.
              spend = charge(spend, texts.length, texts.reduce((sum, text) => sum + text.length, 0))
              spends.set(ctx.sessionID, spend)
              const out: string[] = []
              for (let index = 0; index < texts.length; index += 4) {
                out.push(...(await Promise.all(texts.slice(index, index + 4).map((text) => completeSub([{ role: "user", content: text }])))))
              }
              return out
            },
          }

          // The cheap first pass: one call over previews instead of paying to read
          // the payload. It spends from the same budget as any other sub-call.
          let screenBatchCount = 0
          const screen = makeScreener(completeSub, (chars) => {
            screenBatchCount += 1
            if (screenBatchCount > SCREEN_BATCH_CEILING) {
              throw new Error(
                `screening budget exhausted (${SCREEN_BATCH_CEILING} preview batches per payload). Read the parts you already know about, or ask a narrower question.`,
              )
            }
            // VOLUME only: screening is harness overhead, not one of the questions
            // the trajectory chose to ask, and charging it against that allowance
            // starved them.
            spend = charge(spend, 0, chars)
            spends.set(ctx.sessionID, spend)
          })

          const kernel = yield* Effect.tryPromise({
            try: () => acquire(realmID),
            catch: (err) => new Error(err instanceof Error ? err.message : String(err)),
          }).pipe(Effect.result)
          if (kernel._tag === "Failure") return refuse(`could not start the kernel: ${kernel.failure.message}`)

          const sections: string[] = []
          let payloadChars: number | undefined
          let parts: number | undefined
          let nestedCallCount = 0

          if (params.load) {
            const loaded = yield* Effect.tryPromise({
              try: () => loadPayload(params.load!),
              catch: (err) => new Error(err instanceof Error ? err.message : String(err)),
            }).pipe(Effect.result)
            if (loaded._tag === "Failure") return refuse(loaded.failure.message)
            injectPayload(kernel.success, loaded.success)
            payloadChars = loaded.success.text.length
            parts = loaded.success.partNames.length
            spend = newSpend(params.load.max_subcalls ?? MAX_SUBCALLS_DEFAULT, params.load.max_subcall_chars ?? SUBCALL_CHARS_DEFAULT)
            spends.set(ctx.sessionID, spend)
            sections.push(
              `<kernel loaded="${loaded.success.type}" chars="${payloadChars}" parts="${parts}" subcall_budget="${spend.maxSubcalls}" subcall_chars="${spend.maxSubcallChars}">`,
              `The payload is now a variable. What you print is returned TRUNCATED to a short prefix, so printing a file does not put it in front of you and does not count as reading it: code can tell you a file's name, size and structure, only llm_query can tell you what it MEANS. It begins with:`,
              loaded.success.text.slice(0, PRELUDE_PREFIX_CHARS),
              "</kernel>",
            )
          }

          if (params.code !== undefined) {
            // The nested surface is resolved PER STEP, not once per kernel. It is
            // whatever this request authorised, and a kernel that outlives the
            // request that created it must not keep a stale authorization — nor a
            // stale tool set, since the registry changes with the model and agent.
            const getDefs = toolScriptRegistry.current
            if (!getDefs) return refuse("the tool registry is unavailable, so `tools.*` cannot be resolved")
            const agentInfo = yield* agents.get(ctx.agent)
            const modelHint = ctx.extra?.model as
              | { id: ModelID; providerID: ProviderID; api?: { id: string }; family?: string }
              | undefined
            const defs = (
              yield* getDefs(
                modelHint
                  ? {
                      providerID: modelHint.providerID,
                      modelID: modelHint.id,
                      apiModelID: modelHint.api?.id,
                      family: modelHint.family,
                      agent: agentInfo,
                      harness: ctx.extra?.harness as HarnessMode | undefined,
                    }
                  : undefined,
              )
              // `repl` and `rlm` are excluded on purpose: a nested `repl` would
              // re-enter this same kernel mid-step, and a nested `rlm` would spawn
              // candidate kernels inside a session kernel. Recursion here buys
              // nothing and costs a realm per level.
            ).filter(
              (def) =>
                !TOOL_SCRIPT_EXCLUDED.has(def.id) && def.id !== "repl" && def.id !== "rlm" && def.id !== "toolscript",
            )
            const byId = new Map(defs.map((def) => [def.id, def]))
            const mcpTools = (ctx.extra?.execMcp as { current?: Record<string, AiTool> } | undefined)?.current ?? {}
            const mcpById = new Map(Object.entries(mcpTools).filter(([id]) => !byId.has(id)))
            const toolNames = [...byId.keys(), ...mcpById.keys()].sort()
            kernel.success.set("ALL_TOOLS", toolNames)

            let nestedCalls = 0
            const nestedTrace: string[] = []
            const maxNested = params.max_tool_calls ?? MAX_TOOL_CALLS_DEFAULT
            const callTool = (name: unknown, args: unknown) => {
              const id = String(name)
              const alias = TOOL_SCRIPT_ALIASES[id as keyof typeof TOOL_SCRIPT_ALIASES]
              const def = byId.get(alias ?? id)
              const mcpDef = def ? undefined : mcpById.get(id)
              if (!def && !mcpDef) {
                return Promise.reject(new Error(`unknown tool: ${id}. Available: ${toolNames.join(", ")}`))
              }
              nestedCalls += 1
              if (nestedCalls > maxNested) {
                return Promise.reject(new Error(`tool call budget exceeded (${maxNested} per step)`))
              }
              const seq = nestedCalls
              const started = Date.now()
              const callID = `${ctx.callID ?? "repl"}:${seq}`
              const subCtx: Tool.Context = {
                ...ctx,
                extra: { ...ctx.extra, fromRepl: true },
                callID,
                // Nested metadata is dropped rather than forwarded: writing it to
                // the outer context would replace this step's own title and lose
                // the sibling calls' reporting.
                metadata: () => Effect.void,
              }
              // The map holds the request's WRAPPED executes, so the direct-call
              // pipeline applies unchanged — permission ask, plugin hooks,
              // truncation. The bridge adds a caller, not a second authorization
              // path.
              const run = def
                ? bridge.promise(
                    def.execute(args, subCtx).pipe(Effect.map((result) => ({ title: result.title, output: result.output }))),
                  )
                : bridge.promise(
                    Effect.tryPromise({
                      try: async () =>
                        await mcpDef!.execute!(args ?? {}, { toolCallId: callID, messages: [], abortSignal: ctx.abort }),
                      catch: (err) => (err instanceof Error ? err : new Error(String(err))),
                    }).pipe(
                      Effect.map((result) => {
                        const shaped = result as { title?: string; output?: unknown }
                        return {
                          title: shaped.title ?? id,
                          output: typeof shaped.output === "string" ? shaped.output : JSON.stringify(shaped.output ?? ""),
                        }
                      }),
                    ),
                  )
              return run.then(
                (result) => {
                  nestedTrace.push(`#${seq} ${id} ok ${Date.now() - started}ms`)
                  return result
                },
                (err) => {
                  const message = err instanceof Error ? err.message : String(err)
                  nestedTrace.push(`#${seq} ${id} ERROR ${message.slice(0, 140)}`)
                  throw new Error(`${id}: ${message}`)
                },
              )
            }

            const step = yield* Effect.tryPromise({
              try: () =>
                kernel.success.run(params.code!, {
                  llmQuery: host.llmQuery,
                  llmQueryBatched: host.llmQueryBatched,
                  callTool,
                  screen,
                  interrupt: () => ctx.abort.aborted,
                }),
              catch: (err) => new Error(err instanceof Error ? err.message : String(err)),
            }).pipe(Effect.result)
            // Taken BEFORE the failure check, so a step that spent and then failed
            // still reports the spend in its refusal.
            sideCost = overheadEntries([caller.overhead()])
            if (step._tag === "Failure") return refuse(step.failure.message)
            sections.push(renderStep(step.success).rendered)
            if (nestedTrace.length > 0) sections.push("<tools>", nestedTrace.join("\n"), "</tools>")
            nestedCallCount = nestedCalls
          }

          sections.push(
            `<budget subcalls="${spend.maxSubcalls - spend.subcalls}" chars="${spend.maxSubcallChars - spend.chars}" />`,
          )
          return reply(sections.join("\n"), {
            payloadChars,
            parts,
            nestedCalls: nestedCallCount,
            subcallsLeft: spend.maxSubcalls - spend.subcalls,
            charsLeft: spend.maxSubcallChars - spend.chars,
            subModel: `${subModel.providerID}/${subModel.id}`,
          })
        }),
    }
  }),
)
