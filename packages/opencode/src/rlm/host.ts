/**
 * The wiring a sub-call needs, in one place.
 *
 * Both consumers of the kernel talk to a sub-model — the automated `rlm` loop and
 * the session-native `repl` — and the wiring is not trivial: resolve the lite tier
 * with a fallback, stream with no tools and a replacing system prompt, and account
 * for the tokens. Written twice it drifted the moment one of them changed, so it
 * lives here and each consumer adds only what is its own: `rlm` charges a
 * per-trajectory budget and tallies per candidate, `repl` charges the payload's
 * budget and reports what is left.
 */
import { Effect, Stream } from "effect"
import type { ModelMessage } from "ai"
import { Provider } from "../provider"
import { ModelID, ProviderID } from "../provider/schema"
import { LLM } from "../session/llm"
import { MessageV2 } from "../session/message-v2"
import { EffectBridge } from "@/effect"

/** A sub-model answers about the text it is given and nothing else. Short by
 * design: a long system prompt is repeated on every call and, unlike the
 * trajectory's own history, it is paid for in full every time. */
export const SUB_SYSTEM =
  "You answer a question about the text you are given. Answer only from that evidence, keep every identifier, path and number verbatim, and be concise. If the evidence does not answer the question, say so."

export type SubUsage = { input: number; output: number; cacheRead: number }

/**
 * The screening prompt for a cheap first pass.
 *
 * Why a first pass at all: the only lever that touches the cost floor is HOW MUCH
 * is sent, and a preview index is one to two orders of magnitude smaller than the
 * payload.
 *
 * The bias here was measured, not guessed. The first version said "over-select
 * rather than under-select" and a run on a 462-part payload kept **178 parts — 38 %
 * of it — for 100 % recall and 6 % precision**, i.e. the safety margin was real but
 * it consumed the entire saving. So the instruction now asks for necessity, keeps
 * the asymmetric warning only for genuine uncertainty, and leaves the recall
 * question open to measurement rather than assuming it.
 */
export const SCREEN_PROMPT = `You are given a question and a numbered list of parts of a large document, each shown as a short preview. Decide which parts contain information NEEDED to answer the question.

Be selective. A part belongs on the list only if reading it would materially help answer the question — sharing the same subsystem, or mentioning the same names, is not enough. Listing a part that turns out irrelevant costs a read; omitting one that was needed loses the answer, so include a part only when it is genuinely uncertain AND plausibly decisive.

Reply with the bracketed numbers only, space-separated, and nothing else. If nothing is needed, reply NONE.`

/** How many previews go into one screening call. */
const SCREEN_BATCH = 60

/** Preview characters shown per part. Longer previews sharpen the judgement and
 * cost proportionally more to send; 400 was arbitrary and is now a named
 * constant so it can be varied against measured precision. */
export const SCREEN_PREVIEW_CHARS = 400

/**
 * Ceiling on screening batches per run.
 *
 * Screening is HARNESS overhead: 8 batches on a 462-part payload, per `screen()`
 * call, and a measured run spent 16 of its 21 model calls there. Charging those
 * against the trajectory's own sub-call allowance starved the questions it wanted
 * to ask, so they are charged by VOLUME only and bounded here instead.
 */
export const SCREEN_BATCH_CEILING = 40

/**
 * Which parts are worth reading, judged from previews alone.
 *
 * `completeSub` is the same cheap caller the trajectory already uses, so the
 * saving comes from the SIZE of what is sent, not from a cheaper model — which
 * matters here because the sub-model is already the lite tier.
 */
export const makeScreener =
  (completeSub: (messages: ModelMessage[]) => Promise<string>, charge?: (chars: number) => void) =>
  // Host-hook signature: the guest passes values across the sandbox, so the
  // parameters arrive as `unknown` and are narrowed here.
  async (question: unknown, index: unknown): Promise<number[]> => {
    const asked = String(question)
    const entries = Array.isArray(index) ? (index as Array<{ name: string; preview: string }>) : []
    const relevant = new Set<number>()
    for (let start = 0; start < entries.length; start += SCREEN_BATCH) {
      const slice = entries.slice(start, start + SCREEN_BATCH)
      const listing = slice
        .map((entry, offset) => `[${start + offset}] ${entry.name}\n${entry.preview}`)
        .join("\n---\n")
      const prompt = `${SCREEN_PROMPT}\n\nQuestion: ${asked}\n\nParts:\n${listing}`
      // Screening makes model calls, so it spends from the same allowance as any
      // other: a pass that did not would be a way around the budget.
      charge?.(prompt.length)
      const answer = await completeSub([{ role: "user", content: prompt }])
      for (const found of answer.matchAll(/\[(\d+)\]/g)) {
        const at = Number(found[1])
        if (at >= start && at < start + slice.length) relevant.add(at)
      }
    }
    return [...relevant].sort((a, b) => a - b)
  }

export type SubCaller = {
  completeSub: (messages: ModelMessage[]) => Promise<string>
  subModel: Provider.Model
}

/**
 * Capture the services once, at DEFINITION time, and return a plain-async builder.
 *
 * This shape is forced, not stylistic: a tool's `execute` must have `R = never`,
 * so it cannot yield `Provider`/`LLM` itself. The definition effect may, because
 * the registry provides them — so the services are captured there and the
 * per-request builder is ordinary async, sharing the one bridge.
 */
export const makeSubCallerFactory = Effect.fn("Rlm.subCallerFactory")(function* () {
  const provider = yield* Provider.Service
  const llm = yield* LLM.Service
  const bridge = yield* EffectBridge.make()

  const build = async (input: {
    sessionID: string
    agent: unknown
    user: MessageV2.User
    actorID?: string
    override?: string
    /** Called ONCE per sub-call, with that call's own totals and its index — not
     * per stream step. A per-call figure is what makes cache REUSE measurable: a
     * global ratio cannot say whether a repeat question about the same part was
     * served from the provider's cache, which is the only cache lever this design
     * has. */
    onUsage?: (usage: SubUsage, callIndex: number) => void
  }): Promise<SubCaller> => {
    let callIndex = 0
    // The override is asked for first, then the configured lite tier, then the
    // session's own model — a missing lite tier must not make the tool unusable.
    const overrideModel = input.override
      ? await bridge.promise(
          provider
            .resolveModelRef(input.override, input.user.model.providerID as ProviderID)
            .pipe(Effect.catch(() => Effect.succeed(undefined))),
        )
      : undefined
    const configured = await bridge.promise(
      provider.getSmallModel(input.user.model.providerID as ProviderID).pipe(Effect.catch(() => Effect.succeed(undefined))),
    )
    const root = await bridge.promise(
      provider.getModel(input.user.model.providerID as ProviderID, input.user.model.modelID as ModelID),
    )
    const subModel = overrideModel ?? configured ?? root

    const completeSub = (messages: ModelMessage[]) => {
      const index = callIndex++
      const used: SubUsage = { input: 0, output: 0, cacheRead: 0 }
      return bridge.promise(
        Effect.gen(function* () {
          let text = ""
          const stream = llm.stream({
            user: input.user,
            sessionID: input.sessionID,
            model: subModel,
            agent: input.agent as never,
            system: [],
            // No agent prompt and no tools: the sub-call answers a question about
            // text, and a tool roster would invite it to go looking instead.
            prebuiltSystem: [SUB_SYSTEM],
            messages,
            tools: {},
            toolChoice: "none",
            agentID: input.actorID,
            quietRetryDiagnostics: true,
          })
          yield* Stream.runForEach(stream, (event: LLM.Event) => {
            if (event.type === "text-delta") text += event.text
            else if (event.type === "finish-step") {
              used.input += event.usage.inputTokens ?? 0
              used.output += event.usage.outputTokens ?? 0
              used.cacheRead += event.usage.inputTokenDetails?.cacheReadTokens ?? 0
            } else if (event.type === "error") return Effect.fail(event.error)
            return Effect.void
          })
          input.onUsage?.(used, index)
          return text
        }),
      )
    }

    return { completeSub, subModel }
  }

  return { build }
})
