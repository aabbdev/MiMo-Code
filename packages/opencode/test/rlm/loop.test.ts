import { describe, expect, test } from "bun:test"
import type { ModelMessage } from "ai"
import { acquire, disposeSession } from "../../src/rlm/kernel"
import { injectPayload } from "../../src/rlm/payload"
import { codeBlocks, confidenceReportsOf, outsideCode, parseFinal, runLoop, verdictOf } from "../../src/rlm/loop"

/** A root model that replays a fixed script and records every history it saw,
 * so a test can assert on the history itself and not only on the answer. */
function scripted(replies: string[]) {
  const seen: ModelMessage[][] = []
  let at = 0
  return {
    seen,
    complete: async (messages: ModelMessage[]) => {
      seen.push(messages.map((m) => ({ ...m })))
      const reply = replies[Math.min(at, replies.length - 1)]
      at += 1
      return reply ?? ""
    },
  }
}

const sub = async (messages: ModelMessage[]) => `sub:${String(messages[0]?.content ?? "")}`

async function withKernel<T>(id: string, body: (kernel: Awaited<ReturnType<typeof acquire>>) => Promise<T>) {
  const kernel = await acquire(id)
  try {
    return await body(kernel)
  } finally {
    disposeSession(id)
  }
}

describe("reply parsing", () => {
  test("only fenced blocks whose language can run are returned", () => {
    const reply = ["```repl\nlet a = 1\n```", "```text\nnot code\n```", "```js\nlet b = 2\n```"].join("\n")
    expect(codeBlocks(reply)).toEqual(["let a = 1\n", "let b = 2\n"])
  })

  test("a FINAL answer inside a code block is not an answer", () => {
    // The paper documents models doing exactly this: it is a call the harness
    // never defines, so honouring it would end the run on a variable name.
    const buried = "```repl\nglobalThis.acc = 1\nFINAL_VAR(acc)\n```"
    expect(parseFinal(buried)).toBeUndefined()
    expect(outsideCode(buried).trim()).toBe("")
  })

  test("FINAL and FINAL_VAR are read from prose, newlines and all", () => {
    expect(parseFinal("done\nFINAL(the answer\nspans lines)")).toEqual({ kind: "text", value: "the answer\nspans lines" })
    expect(parseFinal("FINAL_VAR(out)")).toEqual({ kind: "var", value: "out" })
    expect(parseFinal("still thinking")).toBeUndefined()
  })
})

describe("confidence reporting", () => {
  test("reads the JSON object the prompt asks for, wherever the model puts it", () => {
    expect(confidenceReportsOf('{"confidence": 87}')).toEqual([87])
    expect(confidenceReportsOf('thinking about it\n{"confidence": 95.5}\n```repl\n1\n```')).toEqual([95.5])
    expect(confidenceReportsOf("no number here")).toEqual([])
  })

  test("the loop aggregates every step's report in log space", async () => {
    await withKernel("ses-loop-conf", async (kernel) => {
      const model = scripted(['{"confidence": 100}\n```repl\n1\n```', '{"confidence": 50}\nFINAL(done)'])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 4,
        maxSubcalls: 1,
      })
      expect(result.answer).toBe("done")
      // log(1) + log(0.5): the confident step costs nothing, the wavering one is
      // charged — which is the whole reason the paper aggregates in log space.
      expect(result.confidence).toBeCloseTo(Math.log(0.5), 10)
    })
  })

  test("the confidence requirement rides on every step, not only the first", async () => {
    await withKernel("ses-loop-remind", async (kernel) => {
      const model = scripted(["```repl\n1\n```", "```repl\n2\n```", "FINAL(done)"])
      await runLoop({ kernel, completeRoot: model.complete, completeSub: sub, query: "q", maxIterations: 3, maxSubcalls: 1 })
      // Measured: the model follows the rule on its first reply and drops it on the
      // next, so every echo has to carry it or the semantic signal is lost. The
      // budget note is appended after the echo, so find the echo by its content.
      const echo = model.seen[1]!.map((m) => String(m.content)).find((content) => content.includes("REPL OUTPUT"))!
      expect(echo).toContain('"confidence"')
      expect(echo).toContain("not text to continue")
    })
  })

  test("a verdict is read from the reply, which is what makes plurality exact", async () => {
    await withKernel("ses-loop-verdict", async (kernel) => {
      const model = scripted([
        '{"confidence": 90}\n{"verdict": "the contract holds in two places"}\nFINAL(a long detailed answer that no comparison could handle)',
      ])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 3,
        maxSubcalls: 1,
      })
      expect(result.verdict).toBe("the contract holds in two places")
      expect(result.answer).toBe("a long detailed answer that no comparison could handle")
      expect(verdictOf('leading {"verdict": "first"} then {"verdict": "last"}')).toBe("last")
      expect(verdictOf("no verdict here")).toBeUndefined()
    })
  })

  test("the final turn carries the verdict reminder, so exact plurality can fire", async () => {
    await withKernel("ses-loop-verdict-remind", async (kernel) => {
      const seen: ModelMessage[][] = []
      let call = 0
      const complete = async (messages: ModelMessage[]) => {
        seen.push(messages.map((m) => ({ ...m })))
        call += 1
        return call <= 2 ? "```repl\n1\n```" : '"verdict": "x"\nFINAL(done)'
      }
      await runLoop({ kernel, completeRoot: complete, completeSub: sub, query: "q", maxIterations: 2, maxSubcalls: 1 })
      // Measured: with no reminder a full run reported ZERO verdicts. The reminder
      // rides on the FINAL TURN note, which iteration 2 sees.
      const note = seen[1]!.map((m) => String(m.content)).find((content) => content.includes("FINAL TURN"))!
      expect(note).toContain("verdict")
    })
  })

  test("the verdict reminder rides on every step, not only near the iteration limit", async () => {
    await withKernel("ses-loop-verdict-early", async (kernel) => {
      const seen: ModelMessage[][] = []
      let call = 0
      const complete = async (messages: ModelMessage[]) => {
        seen.push(messages.map((m) => ({ ...m })))
        call += 1
        // Finishes on iteration 1 of 20, so the FINAL TURN note (left <= 2) never
        // appears — which is exactly the case that reported no verdict.
        return call === 1 ? "```repl\n1\n```" : '"verdict": "x"\nFINAL(done)'
      }
      const result = await runLoop({
        kernel,
        completeRoot: complete,
        completeSub: sub,
        query: "q",
        maxIterations: 20,
        maxSubcalls: 1,
      })
      expect(result.verdict).toBe("x")
      const echo = seen[1]!.map((m) => String(m.content)).find((content) => content.includes("REPL OUTPUT"))!
      expect(echo).toContain("verdict")
      expect(echo).not.toContain("FINAL TURN")
    })
  })

  test("a model that reports no confidence is silent, not confident", async () => {
    await withKernel("ses-loop-conf-silent", async (kernel) => {
      const model = scripted(["```repl\n1\n```", "FINAL(done)"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 4,
        maxSubcalls: 1,
      })
      expect(result.answer).toBe("done")
      // Undefined, so selection can tell "silent" from "confident" rather than
      // treating a missing report as a perfect score.
      expect(result.confidence).toBeUndefined()
    })
  })
})

describe("runLoop", () => {
  test("a two-iteration run answers, and the payload never reaches the root history", async () => {
    const payload = "SECRET-PAYLOAD-MARKER " + "filler ".repeat(500)
    await withKernel("ses-loop-basic", async (kernel) => {
      // Through the real injection path: `context` is a lazy getter over the parts
      // now, so setting it directly would be a no-op.
      injectPayload(kernel, { text: payload, type: "text block", files: 1, partNames: ["payload"], partTexts: [payload] })
      const model = scripted([
        "```repl\nglobalThis.n = context.split(' ').length; console.log('words', n)\n```",
        "FINAL(it has " + "many" + " words)",
      ])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "how many words?",
        maxIterations: 5,
        maxSubcalls: 5,
      })
      expect(result.answer).toBe("it has many words")
      expect(result.iterations).toBe(2)
      expect(result.stopped).toBeUndefined()
      const history = model.seen[1]!.map((m) => String(m.content)).join("\n")
      expect(history).not.toContain("SECRET-PAYLOAD-MARKER")
      expect(history).toContain("words 502")
    })
  })

  test("what comes back to the root stays bounded no matter how much the REPL printed", async () => {
    await withKernel("ses-loop-bound", async (kernel) => {
      const model = scripted(["```repl\nconsole.log('x'.repeat(400000)); 'ok'\n```", "FINAL(done)"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 3,
        maxSubcalls: 1,
      })
      expect(result.answer).toBe("done")
      // The decisive property: 400k characters were produced, and the root's
      // window grew by a bounded echo instead. Without this the "payload in the
      // environment" design buys nothing.
      const echo = String(model.seen[1]![2]!.content)
      expect(echo.length).toBeLessThan(4400)
      expect(echo).toContain("more characters not shown")
      expect(echo).toMatch(/REPL OUTPUT \(4000\d\d chars/)
      // Framed as an input, not as text to continue: the tag shape scored 0/3 on
      // compliance and invited the model to fabricate an echo of its own.
      expect(echo).toContain("not text to continue")
    })
  })

  test("the guest's llm_query reaches the sub-model and is counted", async () => {
    await withKernel("ses-loop-sub", async (kernel) => {
      const model = scripted([
        "```repl\nglobalThis.a = await llm_query('first'); globalThis.b = await llm_query_batched(['x','y']); console.log(a, b.join(','))\n```",
        "FINAL_VAR(a)",
      ])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 4,
        maxSubcalls: 10,
      })
      expect(result.subcalls).toBe(3)
      expect(result.answer).toBe("sub:first")
    })
  })

  test("FINAL_VAR reads a variable the REPL built", async () => {
    await withKernel("ses-loop-var", async (kernel) => {
      const model = scripted(["```repl\nglobalThis.total = [1,2,3].reduce((a,b) => a+b, 0)\n```", "FINAL_VAR(total)"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 4,
        maxSubcalls: 4,
      })
      expect(result.answer).toBe("6")
    })
  })

  test("an unreadable FINAL_VAR is reported to the model instead of ending the run", async () => {
    await withKernel("ses-loop-badvar", async (kernel) => {
      const model = scripted(["```repl\nglobalThis.real = 7\n```", "FINAL_VAR(missing)", "FINAL_VAR(real)"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 5,
        maxSubcalls: 4,
      })
      expect(result.answer).toBe("7")
      expect(result.iterations).toBe(3)
      // The correction is the LAST message of the third root call, not an echo.
      expect(String(model.seen[2]!.at(-1)!.content)).toContain("could not be read")
    })
  })

  test("the sub-call budget stops a runaway fan-out with an instruction the model can act on", async () => {
    await withKernel("ses-loop-budget", async (kernel) => {
      const model = scripted([
        "```repl\nfor (let i = 0; i < 50; i++) { await llm_query('q' + i) }\n```",
        "FINAL(stopped)",
      ])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 3,
        maxSubcalls: 5,
      })
      expect(result.answer).toBe("stopped")
      expect(result.subcalls).toBe(5)
      // The budget must be the model's problem to solve, in words it can follow.
      const echo = String(model.seen[1]![2]!.content)
      expect(echo).toContain("sub-call budget exhausted")
      expect(echo).toContain("FINAL(")
    })
  })

  test("the question is reachable from the guest as `query`, as the prompt's examples assume", async () => {
    await withKernel("ses-loop-query", async (kernel) => {
      const model = scripted(["```repl\nconsole.log('Q:', query)\n```", "FINAL(ok)"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "how many words does the payload hold?",
        maxIterations: 3,
        maxSubcalls: 1,
      })
      expect(result.answer).toBe("ok")
      expect(String(model.seen[1]![2]!.content)).toContain("how many words does the payload hold?")
    })
  })

  test("the whole-loop time budget ends the run instead of letting steps accumulate", async () => {
    await withKernel("ses-loop-wall", async (kernel) => {
      const slow = async () => {
        await new Promise((resolve) => setTimeout(resolve, 80))
        return "```repl\n1 + 1\n```"
      }
      const result = await runLoop({
        kernel,
        completeRoot: slow,
        completeSub: sub,
        query: "q",
        maxIterations: 20,
        maxSubcalls: 2,
        maxWallMs: 50,
      })
      expect(result.answer).toBe("")
      expect(result.stopped).toContain("time budget")
      // Bounded by the global budget, not by 20 iterations of 60-minute steps.
      expect(result.iterations).toBeLessThan(20)
    })
  })

  test("observed characters are the SHOWN echo, not what the REPL produced", async () => {
    await withKernel("ses-loop-observed", async (kernel) => {
      // 400k characters printed. The grounding ratio must count the ~3 KB the
      // root model actually had in front of it, or a run that never read the
      // payload would look grounded.
      const model = scripted(["```repl\nconsole.log('x'.repeat(400000))\n```", "FINAL(done)"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 3,
        maxSubcalls: 1,
      })
      expect(result.answer).toBe("done")
      expect(result.observedChars).toBeGreaterThan(0)
      expect(result.observedChars).toBeLessThan(4000)
    })
  })

  test("running out of iterations still ends in an answer: the synthesis turn is forced", async () => {
    await withKernel("ses-loop-synth", async (kernel) => {
      const seen: ModelMessage[][] = []
      let call = 0
      const complete = async (messages: ModelMessage[]) => {
        seen.push(messages.map((m) => ({ ...m })))
        call += 1
        // Every budgeted iteration works; the forced synthesis turn answers.
        return call <= 3 ? "```repl\nconsole.log('working')\n```" : "FINAL(assembled from what I gathered)"
      }
      const result = await runLoop({
        kernel,
        completeRoot: complete,
        completeSub: sub,
        query: "q",
        maxIterations: 3,
        maxSubcalls: 2,
      })
      // The payload has already been paid for: returning nothing is the worst
      // outcome this loop can produce, so the budget is not the last word.
      expect(result.answer).toBe("assembled from what I gathered")
      expect(result.iterations).toBe(3)
      expect(result.stopped).toBeUndefined()
      expect(result.trace.join("\n")).toContain("synthesis")
      expect(String(seen.at(-1)!.at(-1)!.content)).toContain("iteration budget is exhausted")
    })
  })

  test("the loop escalates: converge past the threshold, then demand an answer", async () => {
    await withKernel("ses-loop-escalate", async (kernel) => {
      const seen: ModelMessage[][] = []
      let call = 0
      const complete = async (messages: ModelMessage[]) => {
        seen.push(messages.map((m) => ({ ...m })))
        call += 1
        return call <= 10 ? "```repl\n1\n```" : "FINAL(done)"
      }
      const flatten = (messages: ModelMessage[]) => messages.map((m) => String(m.content)).join("\n")
      await runLoop({ kernel, completeRoot: complete, completeSub: sub, query: "q", maxIterations: 10, maxSubcalls: 2 })
      expect(flatten(seen[0]!)).not.toContain("Budget:")
      // seen[i] is the history the (i+1)-th root call saw, so the note emitted by
      // iteration 6 first appears in seen[6] — iteration 7's history.
      expect(flatten(seen[5]!)).not.toContain("Begin converging")
      expect(flatten(seen[6]!)).toContain("Begin converging")
      expect(flatten(seen[8]!)).toContain("FINAL TURN")
    })
  })

  test("if even the forced synthesis yields nothing, the run says so instead of guessing", async () => {
    await withKernel("ses-loop-nosynth", async (kernel) => {
      const result = await runLoop({
        kernel,
        completeRoot: async () => "I am still thinking about the best approach.",
        completeSub: sub,
        query: "q",
        maxIterations: 2,
        maxSubcalls: 1,
      })
      expect(result.answer).toBe("")
      expect(result.stopped).toContain("forced synthesis")
      // The tail of the failing reply is kept, so the cause is visible.
      expect(result.trace.join("\n")).toContain("still thinking")
    })
  })

  test("running out of iterations is reported as stopped, never as an answer", async () => {
    await withKernel("ses-loop-cap", async (kernel) => {
      const model = scripted(["```repl\nconsole.log('tick')\n```"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 3,
        maxSubcalls: 3,
      })
      expect(result.answer).toBe("")
      expect(result.stopped).toContain("3 iterations")
    })
  })

  test("a reply with no runnable block gets a nudge rather than the loop ending", async () => {
    await withKernel("ses-loop-nudge", async (kernel) => {
      const model = scripted(["I will analyse the context now.", "FINAL(done)"])
      const result = await runLoop({
        kernel,
        completeRoot: model.complete,
        completeSub: sub,
        query: "q",
        maxIterations: 4,
        maxSubcalls: 2,
      })
      expect(result.answer).toBe("done")
      expect(String(model.seen[1]![2]!.content)).toContain("No code block was found")
    })
  })
})
