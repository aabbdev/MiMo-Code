/**
 * The RLM loop — Algorithm 1 of arXiv 2512.24601.
 *
 * The property that makes it work is the small one: what returns to the root
 * model's history between iterations is only *constant-size metadata* about what
 * the REPL produced. The payload lives in the kernel as a variable and never
 * enters the window, so the root's context stays bounded no matter how large the
 * input is — the paper's "at most K/c root iterations" instead of a window that
 * fills up. `renderStep` is where that discipline is enforced.
 *
 * The completion call is injected rather than imported: the loop is then
 * exercisable against the real kernel with a scripted model, with no network and
 * no provider.
 */
import type { ModelMessage } from "ai"
import type { Kernel } from "./kernel"
import type { KernelStep } from "./kernel"
import { CONFIDENCE_REMINDER, NUDGE, VERDICT_REMINDER, userTurn } from "./prompt"
import { verbalizedConfidence, firstSentence, splitVerdict } from "./search"

export type Complete = (messages: ModelMessage[]) => Promise<string>

/**
 * A budget shared across nested loops.
 *
 * A nested `rlm_query` must charge its parent's allowance rather than grant itself
 * a fresh one: with independent budgets, one `rlm_query` per level multiplies the
 * ceiling at every level, which is the unbounded-cost failure by construction.
 */
export type Budget = {
  charge: (count: number, chars: number) => void
}

export type LoopInput = {
  /** When given, sub-calls are charged here instead of to this loop's own limits. */
  budget?: Budget
  /** Forwarded to the guest as `rlm_query`, so a nested loop can nest again. */
  rlmQuery?: (text: string, question: string) => Promise<string>
  /** Forwarded to the guest as `screen`: the cheap first pass over the index. */
  screen?: (question: unknown, index: unknown) => Promise<number[]>
  kernel: Kernel
  /** Root turn. The caller owns the system prompt. */
  completeRoot: Complete
  /** One-shot sub-model call, the thing the guest sees as `llm_query`. */
  completeSub: Complete
  query: string
  maxIterations: number
  maxSubcalls: number
  /** Total characters allowed across all sub-calls. The COUNT bound alone is not
   * a cost bound: 40 calls of 400k characters is ~4M tokens of input, so the
   * volume has to be capped on its own. */
  maxSubcallChars?: number
  subcallMaxChars?: number
  /** Wall budget for the whole loop. Defaults to 30 minutes. */
  maxWallMs?: number
  interrupt?: () => boolean
}

export type LoopResult = {
  answer: string
  iterations: number
  subcalls: number
  /** Characters actually handed to sub-models. */
  subcallChars: number
  /** One short line per iteration, for the caller to surface. */
  trace: string[]
/** Characters of REPL output actually SHOWN to the root model. This is the
 * numerator of the grounding ratio, and it is deliberately the shown part rather
 * than what the REPL produced: the echo is truncated, so printing a 2 MB payload
 * is not reading it. A run that answers while this stays tiny was not grounded in
 * the payload, and the caller must be able to see that. */
observedChars: number
/** Set when the loop stopped WITHOUT a final answer, with the reason. */
stopped?: string
/** The trajectory's own verbalized confidence, `Σ log(ν/100) ≤ 0` over every
 * `{"confidence": ν}` it reported. Undefined when it reported none, which is what
 * keeps "silent" distinguishable from "confident" at selection time. */
confidence?: number
/** The one-line verdict given with the final answer, for exact plurality. */
verdict?: string
}

/** The echo budget. The paper's prompt tells the model it "will only be able to
 * see truncated outputs", which is what forces it to route semantics through
 * `llm_query` instead of printing the payload back into its own window. */
const REPL_ECHO_CHARS = 3000
const ERROR_ECHO_CHARS = 1200
/** A sub-LLM is documented to take ~500k characters; stay under it and say so
 * when the guest overshoots, rather than letting the provider reject the call. */
const SUBCALL_MAX_CHARS = 400_000
const SUBCALL_CONCURRENCY = 4
/** Default ceiling on total characters sent to sub-models (~500k tokens), which
 * is what actually bounds the bill. */
const SUBCALL_TOTAL_CHARS = 2_000_000
/** Whole-loop wall budget. The kernel bounds a single iteration, so without an
 * aggregate ceiling the worst case is maxIterations × that step limit — ten
 * hours of unattended model calls from one tool invocation. */
const LOOP_TOTAL_WALL_MS = 30 * 60 * 1000
/** Share of the iteration budget after which the loop stops being neutral about
 * time and tells the model to converge. The paper names this failure without
 * solving it ("distinguishing between a final answer and a thought is brittle
 * for RLMs"); a measured run over a 2.3 MB tree spent its last six iterations
 * flailing AFTER the read had already succeeded — three of them writing no code
 * at all — and never produced a FINAL. */
const CONVERGE_AFTER = 0.6
/** At or below this many remaining iterations, a turn becomes a synthesis
 * request rather than an invitation to explore. */
const FINAL_TURNS = 2

const SYNTHESIS = `The iteration budget is exhausted. You must answer now, from what you have ALREADY gathered in the REPL — do not explore further. Reply with exactly one of: FINAL(your answer), or FINAL_VAR(name) naming a variable that already holds your answer, and give the ${VERDICT_REMINDER} object with it. If you did not manage to gather enough, say so inside FINAL(...) rather than returning nothing.`
/** The final answer is the one thing the harness keeps in full, but an unbounded
 * aggregate would still blow the tool result budget, so bound it here. */
const ANSWER_MAX_CHARS = 100_000

const BLOCK = /```([A-Za-z]*)[ \t]*\r?\n([\s\S]*?)```/g
const RUNNABLE = new Set(["repl", "js", "javascript", "mjs", "ts", "typescript"])

/** Fenced blocks the guest may run. A block whose language is anything else is
 * prose (a ```text quote of the context, say) and running it would be wrong. */
export function codeBlocks(reply: string): string[] {
  return [...reply.matchAll(BLOCK)]
    .filter((m) => RUNNABLE.has((m[1] ?? "").toLowerCase()))
    .map((m) => m[2] ?? "")
    .filter((code) => code.trim().length > 0)
}

/** What the model said OUTSIDE any code block. The paper documents models that
 * bury `FINAL_VAR(x)` inside a ```repl block, where it is a call this harness
 * never defines — treating that as the answer would end the run on a variable
 * name. Only a final answer in prose counts. */
export function outsideCode(reply: string): string {
  return reply.replace(BLOCK, "")
}

/** Every `{"confidence": ν}` a reply reports. The prompt asks for exactly one per
 * step; collecting all of them means a trajectory that volunteers several is
 * charged for each, which is the honest reading of its own self-assessment. */
export function confidenceReportsOf(reply: string): number[] {
  const out: number[] = []
  for (const found of reply.matchAll(/"confidence"\s*:\s*(-?\d+(?:\.\d+)?)/g)) {
    const value = Number(found[1])
    if (Number.isFinite(value)) out.push(value)
  }
  return out
}

/** The one-line verdict a trajectory gives with its final answer. This is what
 * makes the paper's `out(p) = a` exact: comparing 20 000 characters of prose is
 * not a comparison, and a similarity threshold on it is a guess. */
export function verdictOf(reply: string): string | undefined {
  let found: string | undefined
  for (const match of reply.matchAll(/"verdict"\s*:\s*"([^"]{1,400})"/g)) found = match[1]
  const trimmed = found?.trim()
  return trimmed && trimmed.length > 0 ? trimmed : undefined
}

export type Final = { kind: "text"; value: string } | { kind: "var"; value: string }

export function parseFinal(reply: string): Final | undefined {
  const visible = outsideCode(reply)
  const asVar = /FINAL_VAR\s*\(\s*([A-Za-z_$][\w$]*)\s*\)/.exec(visible)
  if (asVar) return { kind: "var", value: asVar[1]! }
  const at = visible.lastIndexOf("FINAL(")
  if (at === -1) return undefined
  const tail = visible.slice(at + "FINAL(".length)
  const close = tail.lastIndexOf(")")
  return { kind: "text", value: (close === -1 ? tail : tail.slice(0, close)).trim() }
}

function stringify(value: unknown): string {
  if (value === undefined) return "(undefined)"
  if (typeof value === "string") return value
  try {
    const json = JSON.stringify(value)
    return json === undefined ? String(value) : json
  } catch {
    return String(value)
  }
}

/**
 * A step's output, framed as an INPUT rather than as a continuation.
 *
 * The format is load-bearing, and it was measured rather than guessed. With the
 * previous `<repl_out …>` wrapper, DeepSeek-V4.1-Flash treated the echo as text to
 * complete — one reply wrote a code block, then fabricated a `<repl_out>` of its
 * own, then copied the trailing reminder back — and it stopped reporting
 * `{"confidence": N}` the moment the history filled with code. Three variants over
 * three trials each: the tag shape scored 0/3 on compliance, an explicit
 * "this is the result of YOUR step, do not repeat it" framing plus the reminder
 * scored 2/3, and a shorter version scored 0/3 — so the framing is what matters,
 * not the brevity.
 */
function renderStep(step: KernelStep): { rendered: string; shown: number } {
  const framing = `This is the result of YOUR previous step, not text to continue. Do not repeat it. ${CONFIDENCE_REMINDER} ${VERDICT_REMINDER}`
  if (step.error) {
    return {
      rendered: `REPL OUTPUT (error, ${step.error.length} chars):\n${step.error.slice(0, ERROR_ECHO_CHARS)}\n\n${framing}`,
      shown: 0,
    }
  }
  const parts: string[] = []
  if (step.value !== undefined) parts.push(`returned: ${stringify(step.value)}`)
  for (const line of step.logs) parts.push(`log: ${line}`)
  const body = parts.join("\n")
  const visible = body.slice(0, REPL_ECHO_CHARS)
  const cut = body.length - visible.length
  return {
    rendered: `REPL OUTPUT (${body.length} chars, ${parts.length} entries):\n${visible}${
      cut > 0 ? `\n… [${cut} more characters not shown — console.log a smaller slice, or use llm_query on the variable]` : ""
    }\n\n${framing}`,
    shown: visible.length,
  }
}

/** Run the root loop until a final answer, or until the iteration or sub-call
 * budget runs out. Both budgets are load-bearing: the paper names "exploding
 * sub-call costs" as the limitation it leaves open, and a loop with no ceiling
 * against a model that calls `llm_query` on every line is exactly that. */
export async function runLoop(input: LoopInput): Promise<LoopResult> {
  const trace: string[] = []
  const startedAt = Date.now()
  /** The trajectory's self-reported confidences, one per step, read as it goes so
   * the value is available on every exit path. */
  const reports: number[] = []
  const totalWall = input.maxWallMs ?? LOOP_TOTAL_WALL_MS
  const remainingMs = () => totalWall - (Date.now() - startedAt)
  let observedChars = 0
  const timedOut = (iterations: number): LoopResult => ({
    answer: "",
    iterations,
    subcalls,
    subcallChars,
    observedChars,
    confidence: verbalizedConfidence(reports),
    trace,
    stopped: `total time budget of ${Math.max(1, Math.round(totalWall / 60000))} minutes exhausted`,
  })
  const perCall = input.subcallMaxChars ?? SUBCALL_MAX_CHARS
  const total = input.maxSubcallChars ?? SUBCALL_TOTAL_CHARS
  let subcalls = 0
  let subcallChars = 0

  const asText = (prompt: unknown) => (typeof prompt === "string" ? prompt : stringify(prompt))

  const spend = (count: number, volume: number) => {
    if (input.budget) {
      // The shared budget enforces the ceiling; the local counters still record
      // what THIS loop spent, so a nested trajectory reports its own cost while the
      // parent's total stays bounded.
      input.budget.charge(count, volume)
      subcalls += count
      subcallChars += volume
      return
    }
    if (subcalls + count > input.maxSubcalls) {
      throw new Error(
        `sub-call budget exhausted (${subcalls} of ${input.maxSubcalls} used). Stop calling llm_query, aggregate what you already have, and reply with FINAL(...).`,
      )
    }
    if (subcallChars + volume > total) {
      throw new Error(
        `sub-call volume budget exhausted (${subcallChars} of ${total} characters already sent). Send fewer or shorter prompts, aggregate what you already have, and reply with FINAL(...).`,
      )
    }
    subcalls += count
    subcallChars += volume
  }

  const ask = async (prompt: unknown) => {
    const text = asText(prompt)
    if (text.length > perCall) {
      throw new Error(
        `prompt is ${text.length} characters, over the ${perCall}-character limit per sub-call. Split it into smaller chunks.`,
      )
    }
    return input.completeSub([{ role: "user", content: text }])
  }

  const llmQuery = async (prompt: unknown) => {
    spend(1, asText(prompt).length)
    return ask(prompt)
  }

  const llmQueryBatched = async (prompts: unknown) => {
    if (!Array.isArray(prompts)) throw new Error("llm_query_batched expects an array of prompt strings")
    // Spend for the whole batch up front: a batch that cannot be afforded must
    // fail before it half-runs, or the guest sees a phantom budget.
    spend(
      prompts.length,
      prompts.reduce<number>((sum, prompt) => sum + asText(prompt).length, 0),
    )
    const out: string[] = []
    for (let i = 0; i < prompts.length; i += SUBCALL_CONCURRENCY) {
      out.push(...(await Promise.all(prompts.slice(i, i + SUBCALL_CONCURRENCY).map((prompt) => ask(prompt)))))
    }
    return out
  }

  const budgetNote = (iteration: number): string | undefined => {
    const left = input.maxIterations - iteration
    if (left <= FINAL_TURNS) {
      return `FINAL TURN — ${left} iteration(s) left. Stop exploring. Reply NOW with FINAL(...) or FINAL_VAR(...) built from what you have already gathered; run code only to assemble the answer. ${VERDICT_REMINDER}`
    }
    if (iteration / input.maxIterations >= CONVERGE_AFTER) {
      return `Budget: iteration ${iteration} of ${input.maxIterations}, ${left} left. Begin converging — gather your buffers and plan to finish with FINAL(...).`
    }
    return undefined
  }

  const messages: ModelMessage[] = [{ role: "user", content: userTurn(input.query) }]
  // The prompt's own worked examples reference `query`, so it must exist as a
  // guest global. Set here rather than by the caller: the loop owns the question,
  // and a missing binding turns the first copied example into a ReferenceError —
  // at the exact moment the paper shows the first decomposition decides the run.
  input.kernel.set("query", input.query)

  for (let iteration = 1; iteration <= input.maxIterations; iteration++) {
    if (input.interrupt?.()) return { answer: "", iterations: iteration - 1, subcalls, subcallChars, observedChars, confidence: verbalizedConfidence(reports), trace, stopped: "cancelled" }
    if (remainingMs() <= 0) return timedOut(iteration - 1)
    const reply = await input.completeRoot(messages)
    reports.push(...confidenceReportsOf(reply))
    messages.push({ role: "assistant", content: reply })

    const final = parseFinal(reply)
    if (final?.kind === "text" && final.value.length > 0) {
      const { verdict, answer } = splitVerdict(final.value)
      trace.push(`#${iteration} FINAL(${final.value.length} chars)`)
      return {
        answer: answer.slice(0, ANSWER_MAX_CHARS),
        // The elicited verdict when a model volunteers the JSON form, then the one
        // riding inside FINAL, then the first sentence: three chances, and the
        // comparison never fails just because the model stayed silent.
        verdict: verdictOf(reply) ?? verdict ?? firstSentence(answer),
        iterations: iteration,
        subcalls,
        subcallChars,
        observedChars,
        confidence: verbalizedConfidence(reports),
        trace,
      }
    }
    if (final?.kind === "var") {
      const step = await input.kernel.run(`return (${final.value})`, {
        llmQuery,
        llmQueryBatched,
        // Never let one step outrun the whole-loop budget; the floor keeps a
        // spent budget from insta-killing the realm instead of ending cleanly.
        wallMs: Math.max(1000, remainingMs()),
        interrupt: input.interrupt,
        rlmQuery: input.rlmQuery ? (text, question) => input.rlmQuery!(String(text), String(question)) : undefined,
        screen: input.screen ? (question, index) => input.screen!(question, index) : undefined,
      })
      if (step.error) {
        // Do not end the run on a name we cannot resolve: tell the model what
        // happened and let it fix the reference.
        trace.push(`#${iteration} FINAL_VAR(${final.value}) unreadable`)
        messages.push({ role: "user", content: `FINAL_VAR(${final.value}) could not be read: ${step.error.slice(0, ERROR_ECHO_CHARS)}` })
        continue
      }
      const answer = stringify(step.value)
      trace.push(`#${iteration} FINAL_VAR(${final.value}) = ${answer.length} chars`)
      return { answer: answer.slice(0, ANSWER_MAX_CHARS), iterations: iteration, subcalls, subcallChars, observedChars, confidence: verbalizedConfidence(reports), trace }
    }

    const blocks = codeBlocks(reply)
    if (blocks.length === 0) {
      trace.push(`#${iteration} no runnable block`)
      messages.push({ role: "user", content: NUDGE })
      const nudgeNote = budgetNote(iteration)
      if (nudgeNote) messages.push({ role: "user", content: nudgeNote })
      continue
    }
    for (const code of blocks) {
      if (remainingMs() <= 0) return timedOut(iteration)
      const step = await input.kernel.run(code, {
        llmQuery,
        llmQueryBatched,
        // Never let one step outrun the whole-loop budget; the floor keeps a
        // spent budget from insta-killing the realm instead of ending cleanly.
        wallMs: Math.max(1000, remainingMs()),
        interrupt: input.interrupt,
        rlmQuery: input.rlmQuery ? (text, question) => input.rlmQuery!(String(text), String(question)) : undefined,
        screen: input.screen ? (question, index) => input.screen!(question, index) : undefined,
      })
      if (input.interrupt?.()) return { answer: "", iterations: iteration, subcalls, subcallChars, observedChars, confidence: verbalizedConfidence(reports), trace, stopped: "cancelled" }
      const { rendered, shown } = renderStep(step)
      observedChars += shown
      trace.push(`#${iteration} ${code.length} chars in → ${step.error ? "error" : "ok"}, ${rendered.length} chars out`)
      // The confidence requirement rides inside the framing of every echo: stated
      // once at the top of the prompt, it is dropped from the second reply onward.
      messages.push({ role: "user", content: rendered })
    }
    const note = budgetNote(iteration)
    if (note) messages.push({ role: "user", content: note })
  }

  // The budget is spent with no answer. One more root call is worth it: the
  // payload has already been paid for in full, and "I read everything and
  // returned nothing" is the worst outcome this loop can produce. This is the
  // strong form of the escalating note above — not a suggestion but the only
  // thing left to reply.
  trace.push(`synthesis turn after ${input.maxIterations} iterations`)
  const synthesis = await input.completeRoot([...messages, { role: "user", content: SYNTHESIS }])
  reports.push(...confidenceReportsOf(synthesis))
  const forced = parseFinal(synthesis)
  if (forced?.kind === "text" && forced.value.length > 0) {
    const { verdict, answer } = splitVerdict(forced.value)
    trace.push(`synthesis produced FINAL(${forced.value.length} chars)`)
    return {
      answer: answer.slice(0, ANSWER_MAX_CHARS),
      verdict: verdictOf(synthesis) ?? verdict ?? firstSentence(answer),
      iterations: input.maxIterations,
      subcalls,
      subcallChars,
      observedChars,
      confidence: verbalizedConfidence(reports),
      trace,
    }
  }
  if (forced?.kind === "var") {
    const step = await input.kernel.run(`return (${forced.value})`, {
      llmQuery,
      llmQueryBatched,
      wallMs: Math.max(1000, remainingMs()),
      interrupt: input.interrupt,
    })
    if (!step.error) {
      const answer = stringify(step.value)
      trace.push(`synthesis produced FINAL_VAR(${forced.value}) = ${answer.length} chars`)
      return { answer: answer.slice(0, ANSWER_MAX_CHARS), iterations: input.maxIterations, subcalls, subcallChars, observedChars, confidence: verbalizedConfidence(reports), trace }
    }
    trace.push(`synthesis FINAL_VAR(${forced.value}) unreadable: ${step.error.slice(0, 120)}`)
  } else if (synthesis.trim().length > 0) {
    // Keep the tail so a caller can see WHY it gave up instead of only that it did.
    trace.push(`synthesis reply started: ${synthesis.trim().slice(0, 160).replace(/\s+/g, " ")}`)
  }

  return {
    answer: "",
    iterations: input.maxIterations,
    subcalls,
    subcallChars,
    observedChars,
    confidence: verbalizedConfidence(reports),
    trace,
    stopped: `no final answer after ${input.maxIterations} iterations and a forced synthesis turn`,
  }
}
