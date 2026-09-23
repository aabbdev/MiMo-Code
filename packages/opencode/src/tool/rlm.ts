/**
 * `rlm` — answer a question about a payload too large to read.
 *
 * The tool is the boundary between the harness and the RLM loop: it loads the
 * payload, resolves the two models the loop needs (a root model for the
 * iteration and a cheaper one for the sub-calls), and turns each completion into
 * a plain async function so `src/rlm/loop.ts` stays free of Effect. Everything
 * that costs money or reads the filesystem is bounded here, not there.
 */
import z from "zod"
import { Effect, Stream } from "effect"
import type { ModelMessage } from "ai"
import { Provider } from "../provider"
import { ModelID, ProviderID } from "../provider/schema"
import { LLM } from "../session/llm"
import { MessageV2 } from "../session/message-v2"
import { Agent } from "../agent/agent"
import { EffectBridge } from "@/effect"
import { acquire, disposeSession } from "../rlm/kernel"
import { makeScreener, makeSubCallerFactory, type SubUsage } from "../rlm/host"
import { chunkText, injectPayload, loadPayload, type Payload } from "../rlm/payload"
import { runLoop, type Budget, type LoopResult } from "../rlm/loop"
import { ALL_SIGNALS, claimsOf, select, type Signals } from "../rlm/search"
import { systemPrompt } from "../rlm/prompt"
import * as Tool from "./tool"
import DESCRIPTION from "./rlm.txt"

const MAX_ITERATIONS_DEFAULT = 10
const MAX_ITERATIONS_CEILING = 40
/** The paper samples K context-interaction programs and selects among them. This
 * is a MAXIMUM: the run starts with one trajectory and only grows on a measured
 * failure mode, so the usual bill is one, not this number. */
const CANDIDATES_DEFAULT = 3
const CANDIDATES_CEILING = 5
/** The paper's default is depth 1 (sub-calls only). Depth 2 lets a trajectory
 * spawn a nested RLM for a sub-task too hard for one call; 3 is its ceiling. */
const DEPTH_DEFAULT = 1
const DEPTH_CEILING = 3
/** Iterations granted to a nested RLM. Small on purpose: depth multiplies work,
 * and the shared budget — not this number — is what bounds the bill. */
const NESTED_ITERATIONS = 4
const MAX_SUBCALLS_DEFAULT = 40
const MAX_SUBCALLS_CEILING = 200
const SUBCALL_CHARS_DEFAULT = 2_000_000
const SUBCALL_CHARS_CEILING = 20_000_000
const PREFIX_CHARS = 1200
const TRACE_TAIL = 12
/** Below this share of the payload observed, an answer that describes the
 * payload was not grounded in it — see the grounding warning in execute(). */
const GROUNDING_FLOOR_PCT = 5

const SUB_SYSTEM =
  "You answer a question about the text you are given. Answer only from that evidence, keep every identifier, path and number verbatim, and be concise. If the evidence does not answer the question, say so."

/** Token counts for one role in the loop, with the cache split the provider
 * reports. The split is not a detail: on Together a cached input token costs
 * $0.006/M against $0.30/M uncached, so a cost reported without it is wrong by
 * up to 50×. */
export type UsageTally = { input: number; output: number; cacheRead: number }

export type Price = { input: number; output: number; cacheRead: number }

/** Cached tokens bill at the cache rate, the remainder at the input rate. */
export function tallyCost(tally: UsageTally, price: Price): number {
  const billable = Math.max(0, tally.input - tally.cacheRead)
  return (billable * price.input + tally.cacheRead * price.cacheRead + tally.output * price.output) / 1_000_000
}

/** Share of input tokens served from the provider's prefix cache. */
export function hitRate(tally: UsageTally): number {
  return tally.input > 0 ? tally.cacheRead / tally.input : 0
}

function sumTallies(a: UsageTally, b: UsageTally): UsageTally {
  return { input: a.input + b.input, output: a.output + b.output, cacheRead: a.cacheRead + b.cacheRead }
}

/**
 * One metadata shape for every outcome. `Tool.define` infers the metadata type
 * from the returned object, so an error return carrying a different key set
 * makes the success return fail to typecheck — and that failure propagates
 * through the registry layer into every consumer.
 */
type RlmMetadata = {
  error?: string
  iterations: number
  subcalls: number
  subcallChars: number
  tokensIn: number
  tokensOut: number
  payloadChars: number
  /** Share of the payload the root loop actually had in front of it. An answer
   * describing the payload at a low share was not grounded in it. */
  observedPct: number
  parts: number
  /** Share of ALL input tokens served from the prefix cache, and the dollars the
   * run actually cost. Sub-calls carry a distinct chunk each, so they can never
   * hit; the root history is append-only and is where caching has to work. */
  cachedPct: number
  costUsd: number
  rootTokensIn: number
  subTokensIn: number
  candidates: number
  chosen: number
  agreed: boolean
  consistencyBasis: string
  /** Whether the search had to grow past one trajectory, and why. */
  escalated: boolean
  escalationReason?: string
  /** Set when the outcome must not be presented as a clean answer. */
  degraded?: string
  selectionBasis: string
  stopped?: string
  subModel?: string
}

function rlmMetadata(over: Partial<RlmMetadata>): RlmMetadata {
  return {
    iterations: 0,
    subcalls: 0,
    subcallChars: 0,
    tokensIn: 0,
    tokensOut: 0,
    payloadChars: 0,
    observedPct: 0,
    parts: 0,
    cachedPct: 0,
    costUsd: 0,
    rootTokensIn: 0,
    subTokensIn: 0,
    candidates: 0,
    chosen: -1,
    agreed: false,
    consistencyBasis: "",
    escalated: false,
    selectionBasis: "",
    ...over,
  }
}

export const RlmTool = Tool.define(
  "rlm",
  Effect.gen(function* () {
    const provider = yield* Provider.Service
    const llm = yield* LLM.Service
    const agents = yield* Agent.Service
    // Captured at definition time: execute must have R = never, so it cannot yield
    // Provider/LLM itself. The shared factory owns the sub-call wiring.
    const subCaller = yield* makeSubCallerFactory()
    return {
      description: DESCRIPTION,
      parameters: z.object({
        query: z.string().describe("The question to answer about the payload."),
        path: z
          .string()
          .optional()
          .describe("File or directory to load as the payload, relative to the worktree (or absolute inside it)."),
        paths: z
          .array(z.string())
          .optional()
          .describe(
            "Several trees to load as ONE payload, for a question that spans them — the usual shape of a payload past the window. Part names are prefixed with the tree so the answer cannot attribute a file to the wrong one.",
          ),
        text: z.string().optional().describe("Payload given inline, when it is not already a file."),
        max_iterations: z
          .number()
          .int()
          .min(1)
          .max(MAX_ITERATIONS_CEILING)
          .optional()
          .describe(`Iteration ceiling for the loop (default ${MAX_ITERATIONS_DEFAULT}).`),
        max_subcalls: z
          .number()
          .int()
          .min(1)
          .max(MAX_SUBCALLS_CEILING)
          .optional()
          .describe(`Ceiling on the number of sub-model calls (default ${MAX_SUBCALLS_DEFAULT}).`),
        max_subcall_chars: z
          .number()
          .int()
          .min(1000)
          .max(SUBCALL_CHARS_CEILING)
          .optional()
          .describe(
            `Total characters allowed across all sub-calls (default ${SUBCALL_CHARS_DEFAULT}). This is the real cost bound: the call count alone is not one.`,
          ),
        candidates: z
          .number()
          .int()
          .min(1)
          .max(CANDIDATES_CEILING)
          .optional()
          .describe(
            `How many trajectories to sample. An EXPLICIT value is a request and is honoured: asking for 3 runs 3. ` +
              `Left unset, it is adaptive — one trajectory, growing to at most ${CANDIDATES_DEFAULT} only if the first shows a measured failure mode (it produced no answer, or it answered without reading the payload).`,
          ),
        depth: z
          .number()
          .int()
          .min(1)
          .max(DEPTH_CEILING)
          .optional()
          .describe(
            `Recursion depth (default ${DEPTH_DEFAULT}). At 1 a trajectory makes sub-CALLS (\`llm_query\`); at 2+ it can call \`rlm_query(text, question)\`, which spawns a whole nested RLM over a sub-context. Nested loops charge the SAME sub-call budget, so depth cannot multiply the bill.`,
          ),
        signals: z
          .array(z.enum(["verdict", "claims", "content", "confidence", "length"]))
          .optional()
          .describe(
            `Which signals the selection may use (default all). Passing a single name reproduces the SRLM paper's ablation — it reports each signal alone against the combination.`,
          ),
        model: z
          .string()
          .optional()
          .describe("Model for the sub-calls. Defaults to the configured lite/small model, else the session's own model."),
      }),
      execute: (
        params: {
          query: string
          path?: string
          paths?: string[]
          text?: string
          max_iterations?: number
          max_subcalls?: number
          max_subcall_chars?: number
          candidates?: number
          depth?: number
          signals?: Array<"verdict" | "claims" | "content" | "confidence" | "length">
          model?: string
        },
        ctx: Tool.Context,
      ) =>
        Effect.gen(function* () {
          const failed = (message: string) => ({
            title: "rlm",
            metadata: rlmMetadata({ error: message }),
            output: `rlm could not run: ${message}`,
          })

          const loaded = yield* Effect.tryPromise({
            try: (): Promise<Payload> => loadPayload({ path: params.path, paths: params.paths, text: params.text }),
            catch: (err) => new Error(err instanceof Error ? err.message : String(err)),
          }).pipe(Effect.result)
          if (loaded._tag === "Failure") return failed(loaded.failure.message)

          // The loop's root turn needs a real user message: LLM.stream reads the
          // turn context, the model variant and the event messageID off it, so a
          // synthetic one would silently lose all three.
          const users = ctx.messages.flatMap((message) => (message.info.role === "user" ? [message.info] : []))
          const user = users[users.length - 1]
          if (!user) return failed("no user message in this session to attribute the sub-calls to")

          const agent = yield* agents.get(ctx.agent)
          const rootModel = yield* provider
            .getModel(user.model.providerID as ProviderID, user.model.modelID as ModelID)
            .pipe(Effect.result)
          if (rootModel._tag === "Failure") return failed(`could not resolve the session's model`)


          const bridge = yield* EffectBridge.make()
          const instances = params.candidates ?? CANDIDATES_DEFAULT
          const payloadChars = loaded.success.text.length
          const partCount = loaded.success.partNames.length
          // The payload's own identifiers, extracted once, so each candidate's
          // grounding is a set membership rather than a scan of 5 MB per claim.
          const known = claimsOf(loaded.success.text)
          const system = systemPrompt({
            type: loaded.success.type,
            length: loaded.success.text.length,
            prefix: loaded.success.text.slice(0, PREFIX_CHARS),
            partCount,
          })
          const realmID = (index: number) => `${ctx.sessionID}#c${index}`
          const maxDepth = params.depth ?? DEPTH_DEFAULT
          const maxSubcalls = params.max_subcalls ?? MAX_SUBCALLS_DEFAULT
          const maxSubcallChars = params.max_subcall_chars ?? SUBCALL_CHARS_DEFAULT
          let chargedCalls = 0
          let chargedChars = 0
          // ONE allowance for the whole run, nested loops included. Without this a
          // nested rlm_query would grant itself a fresh ceiling at every level, which
          // is unbounded cost by construction.
          const budget: Budget = {
            charge: (count, chars) => {
              if (chargedCalls + count > maxSubcalls) {
                throw new Error(
                  `sub-call budget exhausted (${chargedCalls} of ${maxSubcalls} used, nested rlm_query included). Stop calling it, aggregate what you have, and reply with FINAL(...).`,
                )
              }
              if (chargedChars + chars > maxSubcallChars) {
                throw new Error(
                  `sub-call volume budget exhausted (${chargedChars} of ${maxSubcallChars} characters sent, nested rlm_query included). Send fewer or shorter prompts, and reply with FINAL(...).`,
                )
              }
              chargedCalls += count
              chargedChars += chars
            },
          }
          // Every candidate resolves the same sub-model, so the first build's
          // answer is the run's answer; the tally stays per candidate.
          let subModelRef: typeof rootModel.success | undefined
          const zero = (): UsageTally => ({ input: 0, output: 0, cacheRead: 0 })
          // Per-sub-call usage, so cache REUSE is measurable: a global ratio cannot
          // say whether a repeat question about the same part was served from the
          // provider's cache, which is the only cache lever this design has.
          const subCallUsage: SubUsage[] = []
          const signals: Signals = params.signals === undefined ? ALL_SIGNALS : {
            verdict: params.signals.includes("verdict"),
            claims: params.signals.includes("claims"),
            content: params.signals.includes("content"),
            confidence: params.signals.includes("confidence"),
            length: params.signals.includes("length"),
          }
          const priceOf = (model: typeof rootModel.success) => ({
            input: model.cost.input,
            output: model.cost.output,
            cacheRead: model.cost.cache.read,
          })

          type Trajectory = {
            index: number
            result?: LoopResult
            tally: { root: UsageTally; sub: UsageTally }
            /** Screening batches this trajectory paid for, separately from its own
             * sub-calls. */
            screenBatches: { calls: number; chars: number }
            /** Names the cheap first pass decided were worth reading, and how many
             * passes it took. Reported because screening is only worth its calls if
             * it finds the relevant parts: an unmeasured screen is an assumption. */
            screened: string[]
            screenCalls: number
            error?: string
          }

          /**
           * One candidate: its own realm, its own budgets, its own accounting.
           *
           * The isolation is not cosmetic. A realm is stateful — the payload and
           * every variable the trajectory builds live inside it — so two
           * candidates sharing one would have the second start from the first's
           * buffers. That is not an independent sample, and the search rests
           * entirely on having K of them.
           */
          const runOne = async (index: number): Promise<Trajectory> => {
            const tally = { root: zero(), sub: zero() }
            // Root turns stay here — the loop owns that prompt — while sub-calls come
            // from the shared factory, which also tallies them.
            const { completeSub, subModel } = await subCaller.build({
              sessionID: ctx.sessionID,
              agent,
              user,
              actorID: ctx.actorID,
              override: params.model,
              onUsage: (usage) => {
                tally.sub.input += usage.input
                tally.sub.output += usage.output
                tally.sub.cacheRead += usage.cacheRead
                subCallUsage.push(usage)
              },
            })
            subModelRef ??= subModel
            // Counted apart from the loop's sub-calls. Both spend the same budget,
            // but they are different things: a screening batch is harness overhead
            // (8 batches per `screen()` on a 462-part payload), while `subcalls` is
            // what the trajectory chose to ask. Reporting one number for both made
            // "5 sub-calls" understate the model calls by 4x and made a cache ratio
            // compare batches against questions.
            const screenBatches = { calls: 0, chars: 0 }
            const screener = makeScreener(completeSub, (chars) => {
              screenBatches.calls += 1
              screenBatches.chars += chars
              budget.charge(1, chars)
            })
            const screened: string[] = []
            let screenCalls = 0
            const screen = async (question: unknown, index: unknown) => {
              const kept = await screener(question, index)
              screenCalls += 1
              const entries = Array.isArray(index) ? (index as Array<{ name: string }>) : []
              for (const at of kept) if (entries[at]?.name) screened.push(entries[at]!.name)
              return kept
            }
            const completeRoot =
              (model: typeof rootModel.success, systemPromptArray: string[]) =>
              (messages: ModelMessage[]) =>
                bridge.promise(
                  Effect.gen(function* () {
                    let text = ""
                    const stream = llm.stream({
                      user,
                      sessionID: ctx.sessionID,
                      model,
                      agent,
                      system: [],
                      // The RLM prompt replaces the agent prompt entirely: this loop
                      // has no tools and no JSON tool-calling, exactly as the paper
                      // specifies, and the agent's roster would only invite it back.
                      prebuiltSystem: systemPromptArray,
                      messages,
                      tools: {},
                      toolChoice: "none",
                      agentID: ctx.actorID,
                      quietRetryDiagnostics: true,
                    })
                    yield* Stream.runForEach(stream, (event: LLM.Event) => {
                      if (event.type === "text-delta") text += event.text
                      else if (event.type === "finish-step") {
                        tally.root.input += event.usage.inputTokens ?? 0
                        tally.root.output += event.usage.outputTokens ?? 0
                        tally.root.cacheRead += event.usage.inputTokenDetails?.cacheReadTokens ?? 0
                      } else if (event.type === "error") return Effect.fail(event.error)
                      return Effect.void
                    })
                    return text
                  }),
                )

            /**
             * A nested RLM over a sub-context — the paper's depth > 1.
             *
             * Defined here, inside the candidate, because it must reuse THIS
             * trajectory's completions and tally: a nested loop that built its own
             * would double-count its tokens and could resolve a different
             * sub-model. It charges the run's SHARED budget, so nesting cannot
             * multiply the bill, and its realm is dropped the moment it answers.
             */
            let nestedSeq = 0
            const runNested = async (text: string, question: string, level: number): Promise<string> => {
              const nestedID = `${realmID(index)}#d${level}:${nestedSeq++}`
              const child = await acquire(nestedID)
              try {
                const chunks = chunkText("sub-context", text)
                child.set("context", text)
                child.set("context_parts", chunks.map((c) => c.text))
                child.set("context_part_names", chunks.map((c) => c.name))
                const nested = await runLoop({
                  kernel: child,
                  completeRoot: completeRoot(rootModel.success, [
                    systemPrompt({
                      type: "sub-context handed to rlm_query",
                      length: text.length,
                      prefix: text.slice(0, PREFIX_CHARS),
                      partCount: chunks.length,
                    }),
                  ]),
                  completeSub,
                  query: question,
                  maxIterations: NESTED_ITERATIONS,
                  maxSubcalls,
                  maxSubcallChars,
                  budget,
                  interrupt: () => ctx.abort.aborted,
                  rlmQuery: level < maxDepth ? (deeper, asked) => runNested(deeper, asked, level + 1) : undefined,
                })
                return nested.answer.length > 0
                  ? nested.answer
                  : `rlm_query produced no answer: ${nested.stopped ?? "unknown reason"}`
              } finally {
                disposeSession(nestedID)
              }
            }

            const kernel = await acquire(realmID(index))
            injectPayload(kernel, loaded.success)
            const result = await runLoop({
              kernel,
              completeRoot: completeRoot(rootModel.success, [system]),
              completeSub,
              query: params.query,
              maxIterations: params.max_iterations ?? MAX_ITERATIONS_DEFAULT,
              maxSubcalls,
              maxSubcallChars,
              budget,
              screen,
              rlmQuery: maxDepth > 1 ? (text, question) => runNested(text, question, 2) : undefined,
              // A cancelled turn must not be reported as a failed one.
              interrupt: () => ctx.abort.aborted,
            })
            return { index, result, tally, screened, screenCalls, screenBatches }
          }

          /**
           * Whether ONE trajectory is enough.
           *
           * The search exists to discard bad decompositions, and both failure
           * modes measured in this codebase are visible from a single trajectory:
           * answering without reading anything (the name-derived answer), and not
           * answering at all (the trajectory that read everything and never
           * converged). Escalating on "no confidence reported" would buy nothing —
           * every candidate of an uncooperative model lacks it equally, so K more
           * would cost K times as much for the same silence.
           */
          const escalationReason = (result: LoopResult | undefined): string | undefined => {
            if (!result) return "the first trajectory failed to start"
            if (result.answer.trim().length === 0) return `the first trajectory produced no answer (${result.stopped ?? "unknown reason"})`
            const couldNotHaveRead = result.subcalls === 0 && result.observedChars < payloadChars * (GROUNDING_FLOOR_PCT / 100)
            if (couldNotHaveRead) return "the first trajectory answered without reading the payload"
            return undefined
          }

          const runBatch = (indices: number[]) =>
            Effect.promise(async () => {
              try {
                return await Promise.all(
                  indices.map((index) =>
                    runOne(index).catch(
                      (err): Trajectory => ({
                        index,
                        tally: { root: zero(), sub: zero() },
                        screened: [],
                        screenCalls: 0,
                        screenBatches: { calls: 0, chars: 0 },
                        error: err instanceof Error ? err.message : String(err),
                      }),
                    ),
                  ),
                )
              } finally {
                // Nothing reads these realms again: a call re-injects its payload,
                // so holding them idle for the sweeper's 30 minutes is pure waste.
                for (const index of indices) disposeSession(realmID(index))
              }
            })

          // An EXPLICIT `candidates` is a REQUEST, not a ceiling: a caller who asks
          // for three wants three. Only the default is adaptive — treating the
          // explicit value as a maximum (as this did) both broke the knob and made
          // the plurality path impossible to exercise deliberately.
          const requested = params.candidates !== undefined
          let trajectories = yield* runBatch([0])
          const escalation = requested || instances <= 1 ? undefined : escalationReason(trajectories[0]!.result)
          if (requested ? instances > 1 : escalation !== undefined) {
            const rest = Array.from({ length: instances - 1 }, (_, offset) => offset + 1)
            trajectories = [...trajectories, ...(yield* runBatch(rest))]
          }
          const escalated = escalation !== undefined

          const picked = select(
            trajectories.map((trajectory) => ({
              answer: trajectory.result?.answer ?? "",
              verdict: trajectory.result?.verdict,
              confidence: trajectory.result?.confidence,
              // The paper's Len is the trajectory's own generated trace. Root output
              // only: a trajectory that delegates more to sub-models has written
              // less reasoning itself, not more deliberation.
              length: trajectory.tally.root.output,
            })),
            known,
            signals,
          )
          const winner = picked.chosen >= 0 ? trajectories[picked.chosen] : undefined
          const result = winner?.result

          const rootTotals = trajectories.reduce((sum, trajectory) => sumTallies(sum, trajectory.tally.root), zero())
          const subTotals = trajectories.reduce((sum, trajectory) => sumTallies(sum, trajectory.tally.sub), zero())
          const totals = sumTallies(rootTotals, subTotals)
          // The bill is for all K trajectories, not just the one that won — the
          // search is what was paid for.
          const pricedSub = subModelRef ?? rootModel.success
          const costUsd = tallyCost(rootTotals, priceOf(rootModel.success)) + tallyCost(subTotals, priceOf(pricedSub))
          const cachedPct = hitRate(totals) * 100
          // Reported separately because the two halves have opposite cache
          // economics, and the total alone reads like a failure: sub-calls carry a
          // distinct chunk each and can never hit, so they drag the average down
          // no matter how well the root prefix caches.
          const rootCachedPct = hitRate(rootTotals) * 100
          const iterations = trajectories.reduce((sum, trajectory) => sum + (trajectory.result?.iterations ?? 0), 0)
          const subcalls = trajectories.reduce((sum, trajectory) => sum + (trajectory.result?.subcalls ?? 0), 0)
          const subcallChars = trajectories.reduce((sum, trajectory) => sum + (trajectory.result?.subcallChars ?? 0), 0)
          const observedPct = payloadChars === 0 ? 100 : ((result?.observedChars ?? 0) / payloadChars) * 100

          // What the cheap first pass kept, across every trajectory. A recall
          // measurement needs the names, not a count: the count cannot say whether
          // the parts that mattered were among them.
          const screenedNames = [...new Set(trajectories.flatMap((trajectory) => trajectory.screened))].sort()
          const screenCalls = trajectories.reduce((sum, trajectory) => sum + trajectory.screenCalls, 0)
          const screenModelCalls = trajectories.reduce((sum, trajectory) => sum + trajectory.screenBatches.calls, 0)
          const screenModelChars = trajectories.reduce((sum, trajectory) => sum + trajectory.screenBatches.chars, 0)
          const screening =
            screenCalls === 0
              ? ""
              : `<screening calls="${screenCalls}" batches="${screenModelCalls}" batch_chars="${screenModelChars}" kept="${screenedNames.length}" of="${partCount}">\n${screenedNames.join("\n")}\n</screening>`
          const tableau = picked.report
            .map((row) => {
              const failed = trajectories[row.index]?.error
              return [
                `#${row.index}`,
                row.eligible ? "answering" : "no-answer",
                `len=${row.length}`,
                `conf=${row.confidence === undefined ? "none" : row.confidence.toFixed(2)}`,
                `score=${row.score === undefined ? "none" : row.score.toFixed(0)}`,
                `claims=${row.claims}`,
                `ground=${row.grounding === undefined ? "n/a" : `${Math.round(row.grounding * 100)}%`}`,
                `"${row.preview}"`,
                failed ? `ERROR ${failed}` : "",
                row.index === picked.chosen ? "<-- chosen" : "",
              ]
                .filter(Boolean)
                .join(" ")
            })
            .join("\n")
          // The matrix is what makes `agreed` checkable. Losing answers are not
          // returned — they would blow the result budget — so without this, a claim
          // that two trajectories agreed would be an assertion with nothing behind
          // it, which is the defect this whole accounting exists to remove.
          const matrix =
            trajectories.length > 1
              ? picked.overlap.map((row, index) => `${index}: ${row.map((value) => value.toFixed(2)).join(" ")}`).join("\n")
              : ""

          const stats = [
            `candidates="${instances}"`,
            `chosen="${picked.chosen}"`,
            `agreed="${picked.agreed}"`,
            `iterations="${iterations}"`,
            `subcalls="${subcalls}"`,
            `subcall_chars="${subcallChars}"`,
            `tokens_in="${totals.input}"`,
            `tokens_out="${totals.output}"`,
            `cached_pct="${cachedPct.toFixed(1)}"`,
            `root_cached_pct="${rootCachedPct.toFixed(1)}"`,
            `sub_cached_pct="${(hitRate(subTotals) * 100).toFixed(1)}"`,
            `screen_calls="${screenModelCalls}"`,
            `model_calls="${subCallUsage.length}"`,
            `model_calls_cached="${subCallUsage.filter((usage) => usage.cacheRead > 0).length}/${subCallUsage.length}"`,
            `signals="${params.signals?.join(",") ?? "all"}"`,
            `cost_usd="${costUsd.toFixed(4)}"`,
            `payload_chars="${payloadChars}"`,
            `parts="${partCount}"`,
            `observed_pct="${observedPct.toFixed(1)}"`,
          ].join(" ")
          const trace = (result?.trace ?? []).slice(-TRACE_TAIL).join("\n")
          // The failure this accounting exists to expose: a run that "succeeds",
          // answers confidently about every file, and never says it had seen
          // 0.7 % of them. The echo bound makes printing a payload cheap and
          // reading it impossible, so the two must not be conflated.
          const ungrounded = (result?.answer.length ?? 0) > 0 && result?.subcalls === 0 && observedPct < GROUNDING_FLOOR_PCT
          const warning = ungrounded
            ? `<grounding_warning>\nThis answer was produced after observing only ${observedPct.toFixed(1)}% of the payload, with ZERO sub-calls. Code can report a file's name, size and structure but never its meaning — treat every claim about content as unverified, and re-run with a query that requires reading.\n</grounding_warning>`
            : ""
          const status =
            picked.chosen < 0
              ? trajectories.some((trajectory) => trajectory.result?.stopped === "cancelled")
                ? "cancelled"
                : "stopped"
              : picked.degraded
                ? "degraded"
                : "ok"
          const body =
            picked.chosen >= 0
              ? result!.answer
              : `no candidate produced an answer: ${
                  result?.stopped ?? trajectories.find((trajectory) => trajectory.error)?.error ?? "every trajectory stopped without a final answer"
                }`
          const output = [
            `<rlm status="${status}" ${stats}>`,
            body,
            ...(picked.degraded ? ["<degraded>", picked.degraded, "</degraded>"] : []),
            `<candidates basis="${picked.consistencyBasis}" agreed="${picked.agreed}" consistent="${picked.consistent.join(",")}" substantive="${picked.substantive.join(",")}" escalated="${escalated}"${escalation === undefined ? "" : ` escalation_reason="${escalation}"`}>`,
            tableau,
            ...(matrix ? ["<overlap>", matrix, "</overlap>"] : []),
            "</candidates>",
            ...(screening ? [screening] : []),
            ...(trace ? ["<trace>", trace, "</trace>"] : []),
            `<selection basis="${picked.basis}" />`,
            ...(warning ? [warning] : []),
            "</rlm>",
          ].join("\n")

          return {
            title: `rlm: ${params.query.slice(0, 60)}`,
            metadata: rlmMetadata({
              iterations,
              subcalls,
              subcallChars,
              tokensIn: totals.input,
              tokensOut: totals.output,
              payloadChars,
              observedPct: Number(observedPct.toFixed(1)),
              parts: partCount,
              cachedPct: Number(cachedPct.toFixed(1)),
              costUsd: Number(costUsd.toFixed(4)),
              rootTokensIn: rootTotals.input,
              subTokensIn: subTotals.input,
              candidates: trajectories.length,
              chosen: picked.chosen,
              agreed: picked.agreed,
              consistencyBasis: picked.consistencyBasis,
              escalated,
              escalationReason: escalation,
              degraded: picked.degraded,
              selectionBasis: picked.basis,
              stopped: result?.stopped,
              subModel: `${(subModelRef ?? rootModel.success).providerID}/${(subModelRef ?? rootModel.success).id}`,
            }),
            output,
          }
        }),
    }
  }),
)
