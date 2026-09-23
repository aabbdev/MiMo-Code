import { describe, expect, test } from "bun:test"
import { acquire, autoReturn, declareGlobals, disposeSession } from "../../src/rlm/kernel"

const noop = async () => ""

describe("autoReturn", () => {
  test("a bare trailing expression becomes the step's value", () => {
    expect(autoReturn("1 + 1")).toBe("return (1 + 1)")
    expect(autoReturn("const a = 1\ncontext.length")).toBe("const a = 1\nreturn (context.length)")
  })

  test("a trailing semicolon stays outside the wrapped expression", () => {
    expect(autoReturn("context.length;")).toBe("return (context.length);")
  })

  test("a non-expression tail is left alone", () => {
    expect(autoReturn("if (x) { y() }")).toBe("if (x) { y() }")
    expect(autoReturn("return 1")).toBe("return 1")
  })
})

describe("declareGlobals", () => {
  test("top-level declarations become globals without moving lines", () => {
    const input = ["const a = 1", "let b = 2", "var c = 3"].join("\n")
    expect(declareGlobals(input)).toBe(["globalThis.a = 1", "globalThis.b = 2", "globalThis.c = 3"].join("\n"))
  })

  test("a multi-line initializer keeps its shape so diagnostics stay on the caller's line", () => {
    const input = ["const payload = [", "  1,", "  2,", "].length", "const after = payload"].join("\n")
    const out = declareGlobals(input).split("\n")
    expect(out).toHaveLength(5)
    expect(out[0]).toBe("globalThis.payload = [")
    expect(out[3]).toBe("].length")
    expect(out[4]).toBe("globalThis.after = payload")
  })

  test("a declared type is dropped, since `globalThis.n: number = 1` is not valid", () => {
    expect(declareGlobals("const n: number = 1")).toBe("globalThis.n = 1")
  })

  test("function and class declarations are bound too", () => {
    expect(declareGlobals("function f() { return 1 }")).toBe("globalThis.f = function f() { return 1 };")
    expect(declareGlobals("class C {}")).toBe("globalThis.C = class C {};")
  })

  test("a destructuring declaration is left alone rather than rewritten", () => {
    const input = "const { a, b } = obj"
    expect(declareGlobals(input)).toBe(input)
  })

  test("nested declarations are untouched — only the top level persists", () => {
    const input = "for (const x of [1]) { const y = x }"
    expect(declareGlobals(input)).toBe(input)
  })
})

describe("persistent kernel", () => {
  test("a variable declared in one call is visible in the next", async () => {
    const id = "ses-rlm-persist"
    const kernel = await acquire(id)
    try {
      const first = await kernel.run("const answer = 41", { llmQuery: noop, llmQueryBatched: noop })
      expect(first.error).toBeUndefined()
      const second = await kernel.run("globalThis.out = answer + 1", { llmQuery: noop, llmQueryBatched: noop })
      expect(second.error).toBeUndefined()
      expect(second.value).toBe(42)
    } finally {
      disposeSession(id)
    }
  })

  test("the payload injected by the host is reachable and survives every call", async () => {
    const id = "ses-rlm-payload"
    const kernel = await acquire(id)
    try {
      kernel.set("context", "alpha\nbeta\ngamma")
      const step = await kernel.run("globalThis.lines = context.split('\\n'); lines.length", {
        llmQuery: noop,
        llmQueryBatched: noop,
      })
      expect(step.value).toBe(3)
      const again = await kernel.run("lines[1]", { llmQuery: noop, llmQueryBatched: noop })
      expect(again.value).toBe("beta")
    } finally {
      disposeSession(id)
    }
  })

  test("an async host hook can be awaited, repeatedly, in the same realm", async () => {
    const id = "ses-rlm-llm"
    const calls: string[] = []
    const kernel = await acquire(id)
    const llmQuery = async (prompt: unknown) => {
      calls.push(String(prompt))
      return `answer:${prompt}`
    }
    try {
      const first = await kernel.run("const a = await llm_query('one'); a", { llmQuery, llmQueryBatched: noop })
      expect(first.error).toBeUndefined()
      expect(first.value).toBe("answer:one")
      const second = await kernel.run("const b = await llm_query('two'); a + '|' + b", { llmQuery, llmQueryBatched: noop })
      expect(second.error).toBeUndefined()
      expect(second.value).toBe("answer:one|answer:two")
      expect(calls).toEqual(["one", "two"])
    } finally {
      disposeSession(id)
    }
  })

  test("llm_query_batched reaches the host with every prompt", async () => {
    const id = "ses-rlm-batch"
    let seen: string[] = []
    const kernel = await acquire(id)
    try {
      const step = await kernel.run("const out = await llm_query_batched(['a','b','c']); out.join('-')", {
        llmQuery: noop,
        llmQueryBatched: async (prompts: unknown) => {
          seen = prompts as string[]
          return (prompts as string[]).map((p) => p.toUpperCase())
        },
      })
      expect(step.error).toBeUndefined()
      expect(step.value).toBe("A-B-C")
      expect(seen).toEqual(["a", "b", "c"])
    } finally {
      disposeSession(id)
    }
  })

  test("console output is captured for the caller instead of vanishing", async () => {
    const id = "ses-rlm-log"
    const kernel = await acquire(id)
    try {
      const step = await kernel.run("console.log('hello', { n: 2 })", { llmQuery: noop, llmQueryBatched: noop })
      expect(step.logs).toEqual(["hello {\"n\":2}"])
    } finally {
      disposeSession(id)
    }
  })

  test("a guest error is reported as text and the realm stays usable", async () => {
    const id = "ses-rlm-error"
    const kernel = await acquire(id)
    try {
      const bad = await kernel.run("throw new Error('boom')", { llmQuery: noop, llmQueryBatched: noop })
      expect(bad.error).toContain("boom")
      const after = await kernel.run("1 + 1", { llmQuery: noop, llmQueryBatched: noop })
      expect(after.value).toBe(2)
    } finally {
      disposeSession(id)
    }
  })

  test("tools.<name> reaches the host's bridge, by property and by index", async () => {
    const id = "ses-kernel-bridge"
    const kernel = await acquire(id)
    const calls: Array<[string, unknown]> = []
    const callTool = async (name: unknown, args: unknown) => {
      calls.push([String(name), args])
      return { output: `ran:${String(name)}` }
    }
    try {
      const step = await kernel.run(
        "const a = await tools.read({ file_path: 'src/x.cpp' }); const b = await tools['grep']({ pattern: 'x' }); console.log(a.output, b.output)",
        { llmQuery: noop, llmQueryBatched: noop, callTool },
      )
      expect(step.error).toBeUndefined()
      expect(calls.map(([name]) => name)).toEqual(["read", "grep"])
      expect(calls[0]![1]).toEqual({ file_path: "src/x.cpp" })
      expect(step.logs).toEqual(["ran:read ran:grep"])
    } finally {
      disposeSession(id)
    }
  })

  test("a kernel with no bridge refuses the call in words, instead of failing obscurely", async () => {
    const id = "ses-kernel-nobridge"
    const kernel = await acquire(id)
    try {
      const step = await kernel.run("await tools.read({})", { llmQuery: noop, llmQueryBatched: noop })
      expect(step.error).toContain("no tool bridge")
    } finally {
      disposeSession(id)
    }
  })

  test("a host rejection arrives in the guest as a message it can act on", async () => {
    const id = "ses-kernel-bridge-err"
    const kernel = await acquire(id)
    try {
      const step = await kernel.run("try { await tools.write({}) } catch (e) { console.log('caught:', e.message) }", {
        llmQuery: noop,
        llmQueryBatched: noop,
        callTool: async () => {
          throw new Error("permission denied: write")
        },
      })
      expect(step.error).toBeUndefined()
      expect(step.logs.join(" ")).toContain("permission denied: write")
    } finally {
      disposeSession(id)
    }
  })

  test("ask_about puts the part first, which is what makes a repeat question cacheable", async () => {
    const id = "ses-kernel-askabout"
    const kernel = await acquire(id)
    const prompts: string[] = []
    try {
      kernel.set("context_parts", ["PART-ONE-CONTENT", "PART-TWO-CONTENT"])
      const step = await kernel.run(
        "const a = await ask_about(0, 'what is it?'); const b = await ask_about(0, 'how big?'); console.log(a, b)",
        {
          llmQuery: async (prompt) => {
            prompts.push(String(prompt))
            return "ok"
          },
          llmQueryBatched: noop,
        },
      )
      expect(step.error).toBeUndefined()
      expect(prompts[0]).toBe("PART-ONE-CONTENT\n\nwhat is it?")
      expect(prompts[1]).toBe("PART-ONE-CONTENT\n\nhow big?")
      // The shared PREFIX is the whole point: two different questions about one
      // part re-send the part, and the provider can serve it from cache.
      expect(prompts[0]!.startsWith("PART-ONE-CONTENT")).toBe(true)
      expect(prompts[1]!.startsWith("PART-ONE-CONTENT")).toBe(true)
    } finally {
      disposeSession(id)
    }
  })

  test("ask_about without a payload says so, rather than failing obscurely", async () => {
    const id = "ses-kernel-askabout-empty"
    const kernel = await acquire(id)
    try {
      const step = await kernel.run("try { await ask_about(0, 'x') } catch (e) { console.log(e.message) }", {
        llmQuery: noop,
        llmQueryBatched: noop,
      })
      expect(step.error).toBeUndefined()
      expect(step.logs.join(" ")).toContain("no payload is loaded")
    } finally {
      disposeSession(id)
    }
  })

  test("rlm_query reaches the host, and is refused when the depth forbids it", async () => {
    const id = "ses-kernel-nested"
    const kernel = await acquire(id)
    const seen: string[] = []
    try {
      const step = await kernel.run("const a = await rlm_query('SUB-CONTEXT', 'what is it?'); console.log(a)", {
        llmQuery: noop,
        llmQueryBatched: noop,
        rlmQuery: async (text, question) => {
          seen.push(`${String(text)}|${String(question)}`)
          return "nested answer"
        },
      })
      expect(step.error).toBeUndefined()
      expect(step.logs).toEqual(["nested answer"])
      expect(seen).toEqual(["SUB-CONTEXT|what is it?"])

      // Bound only when the caller allows that depth; otherwise the guest is told so
      // in words rather than hitting an undefined function.
      const refused = await kernel.run("try { await rlm_query('x', 'y') } catch (e) { console.log(e.message) }", {
        llmQuery: noop,
        llmQueryBatched: noop,
      })
      expect(refused.error).toBeUndefined()
      expect(refused.logs.join(" ")).toContain("disabled at this depth")
    } finally {
      disposeSession(id)
    }
  })

  test("two sessions do not share a realm, and one session's death does not touch the other", async () => {
    const a = "ses-rlm-a"
    const b = "ses-rlm-b"
    const kernelA = await acquire(a)
    const kernelB = await acquire(b)
    await kernelA.run("const mine = 'A'", { llmQuery: noop, llmQueryBatched: noop })
    await kernelB.run("const mine = 'B'", { llmQuery: noop, llmQueryBatched: noop })
    expect((await kernelA.run("mine", { llmQuery: noop, llmQueryBatched: noop })).value).toBe("A")
    expect((await kernelB.run("mine", { llmQuery: noop, llmQueryBatched: noop })).value).toBe("B")

    // A step that blows its deadline is unrecoverable and drops ITS realm only.
    const stalled = "__never"
    kernelA.set(stalled, undefined)
    const blown = await kernelA.run(`await new Promise(() => {})`, {
      llmQuery: noop,
      llmQueryBatched: noop,
      wallMs: 60,
    })
    expect(blown.error).toContain("wall-clock")

    const rebuilt = await acquire(a)
    expect((await rebuilt.run("typeof mine", { llmQuery: noop, llmQueryBatched: noop })).value).toBe("undefined")
    expect((await kernelB.run("mine", { llmQuery: noop, llmQueryBatched: noop })).value).toBe("B")
    disposeSession(a)
    disposeSession(b)
  }, 15000)
})
