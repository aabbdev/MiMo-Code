import { describe, test, expect } from "bun:test"
import { NodeFileSystem } from "@effect/platform-node"
import { Effect, FileSystem, Layer } from "effect"
import { Truncate } from "../../src/tool"
import { Identifier } from "../../src/id/id"
import { Process } from "../../src/util"
import { Filesystem } from "../../src/util"
import path from "path"
import { testEffect } from "../lib/effect"
import { writeFileStringScoped } from "../lib/filesystem"

const FIXTURES_DIR = path.join(import.meta.dir, "fixtures")
const ROOT = path.resolve(import.meta.dir, "..", "..")

const it = testEffect(Layer.mergeAll(Truncate.defaultLayer, NodeFileSystem.layer))

describe("Truncate", () => {
  describe("output", () => {
    it.live("truncates large json file by bytes", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const content = yield* Effect.promise(() => Filesystem.readText(path.join(FIXTURES_DIR, "models-api.json")))
        const result = yield* svc.output(content)

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("truncated...")
        if (result.truncated) expect(result.outputPath).toBeDefined()
      }),
    )

    it.live("returns content unchanged when under limits", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const content = "line1\nline2\nline3"
        const result = yield* svc.output(content)

        expect(result.truncated).toBe(false)
        expect(result.content).toBe(content)
      }),
    )

    it.live("truncates by line count", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const lines = Array.from({ length: 100 }, (_, i) => `line${i}`).join("\n")
        const result = yield* svc.output(lines, { maxLines: 10 })

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("...90 lines truncated...")
      }),
    )

    it.live("truncates by byte count", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const content = "a".repeat(1000)
        const result = yield* svc.output(content, { maxBytes: 100 })

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("truncated...")
      }),
    )

    it.live("truncates from head by default", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const lines = Array.from({ length: 10 }, (_, i) => `line${i}`).join("\n")
        const result = yield* svc.output(lines, { maxLines: 3 })

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("line0")
        expect(result.content).toContain("line1")
        expect(result.content).toContain("line2")
        expect(result.content).not.toContain("line9")
      }),
    )

    it.live("truncates from tail when direction is tail", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const lines = Array.from({ length: 10 }, (_, i) => `line${i}`).join("\n")
        const result = yield* svc.output(lines, { maxLines: 3, direction: "tail" })

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("line7")
        expect(result.content).toContain("line8")
        expect(result.content).toContain("line9")
        expect(result.content).not.toContain("line0")
      }),
    )

    test("uses default MAX_LINES and MAX_BYTES", () => {
      expect(Truncate.MAX_LINES).toBe(2000)
      expect(Truncate.MAX_BYTES).toBe(50 * 1024)
    })

    it.live("large single-line file truncates with byte message", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const content = yield* Effect.promise(() => Filesystem.readText(path.join(FIXTURES_DIR, "models-api.json")))
        const result = yield* svc.output(content)

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("bytes truncated...")
        expect(Buffer.byteLength(content, "utf-8")).toBeGreaterThan(Truncate.MAX_BYTES)
      }),
    )

    it.live("writes full output to file when truncated", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const lines = Array.from({ length: 100 }, (_, i) => `line${i}`).join("\n")
        const result = yield* svc.output(lines, { maxLines: 10 })

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("The tool call succeeded but the output was truncated")
        expect(result.content).toContain("Grep")
        if (!result.truncated) throw new Error("expected truncated")
        expect(result.outputPath).toBeDefined()
        expect(result.outputPath).toContain("tool_")

        const written = yield* Effect.promise(() => Filesystem.readText(result.outputPath!))
        expect(written).toBe(lines)
      }),
    )

    it.live("labels truncated error output as failed", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const content = Array.from({ length: 100 }, (_, i) => `error line ${i}`).join("\n")
        const result = yield* svc.output(content, { maxLines: 10, outcome: "error" })

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("The tool call failed but the output was truncated")
        expect(result.content).not.toContain("The tool call succeeded")
      }),
    )

    it.live("suggests actor tool when agent has actor permission", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const lines = Array.from({ length: 100 }, (_, i) => `line${i}`).join("\n")
        const agent = { permission: [{ permission: "actor", pattern: "*", action: "allow" as const }] }
        const result = yield* svc.output(lines, { maxLines: 10 }, agent as any)

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("Grep")
        expect(result.content).toContain("actor tool")
      }),
    )

    it.live("omits actor tool hint when agent lacks actor permission", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const lines = Array.from({ length: 100 }, (_, i) => `line${i}`).join("\n")
        const agent = { permission: [{ permission: "actor", pattern: "*", action: "deny" as const }] }
        const result = yield* svc.output(lines, { maxLines: 10 }, agent as any)

        expect(result.truncated).toBe(true)
        expect(result.content).toContain("Grep")
        expect(result.content).not.toContain("actor tool")
      }),
    )

    it.live("does not write file when not truncated", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const content = "short content"
        const result = yield* svc.output(content)

        expect(result.truncated).toBe(false)
        if (result.truncated) throw new Error("expected not truncated")
        expect("outputPath" in result).toBe(false)
      }),
    )

    test("loads truncate effect in a fresh process", async () => {
      const out = await Process.run([process.execPath, "run", path.join(ROOT, "src", "tool", "truncate.ts")], {
        cwd: ROOT,
      })

      expect(out.code).toBe(0)
    }, 20000)

    it.live("head+tail with errors in tail shows head and tail sections", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        // Build a large output: many normal lines + error lines at the end
        const normalLines = Array.from({ length: 100 }, (_, i) => `normal line ${i}`)
        const errorLines = ["Error: something went wrong", "exit code 1"]
        const allLines = [...normalLines, ...errorLines]
        const text = allLines.join("\n")

        const result = yield* svc.output(text, { maxLines: 10, direction: "head+tail" })

        expect(result.truncated).toBe(true)
        // Should contain head content
        expect(result.content).toContain("normal line 0")
        // Should contain tail error content
        expect(result.content).toContain("Error: something went wrong")
        expect(result.content).toContain("exit code 1")
        // Should contain omission marker
        expect(result.content).toContain("lines omitted — showing head and tail")
      }),
    )

    it.live("head+tail without errors in tail degrades to head mode", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        // Build output with no error keywords at the end
        const lines = Array.from({ length: 100 }, (_, i) => `normal line ${i}`)
        const text = lines.join("\n")

        const result = yield* svc.output(text, { maxLines: 10, direction: "head+tail" })

        expect(result.truncated).toBe(true)
        // Should behave like head: contains first lines
        expect(result.content).toContain("normal line 0")
        // Should NOT contain head+tail omission marker (degraded to head)
        expect(result.content).not.toContain("lines omitted — showing head and tail")
        // Should contain normal truncation marker
        expect(result.content).toContain("truncated...")
      }),
    )

    it.live("pressureCaps halves maxLines before size check", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        // With maxLines=20 and pressureCaps=true, effective limit becomes 10
        const lines = Array.from({ length: 15 }, (_, i) => `line${i}`).join("\n")

        // Without pressureCaps: 15 lines fits within maxLines=20, so not truncated
        const resultNoPressure = yield* svc.output(lines, { maxLines: 20 })
        expect(resultNoPressure.truncated).toBe(false)

        // With pressureCaps: effective maxLines=10, so 15 lines gets truncated
        const resultWithPressure = yield* svc.output(lines, { maxLines: 20, pressureCaps: true })
        expect(resultWithPressure.truncated).toBe(true)
      }),
    )
  })

  describe("cleanup", () => {
    const DAY_MS = 24 * 60 * 60 * 1000

    it.live("deletes files older than 7 days and preserves recent files", () =>
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const fs = yield* FileSystem.FileSystem

        yield* fs.makeDirectory(Truncate.DIR, { recursive: true })

        const v1Name = (ts: number) => {
          const hex = ((BigInt(ts) * 0x1000n + 1n) & 0xffffffffffffn).toString(16).padStart(12, "0")
          return `tool_${hex}${"0".repeat(14)}`
        }
        const old = path.join(Truncate.DIR, Identifier.create("tool", "ascending", Date.now() - 10 * DAY_MS))
        const recent = path.join(Truncate.DIR, Identifier.create("tool", "ascending", Date.now() - 3 * DAY_MS))
        const oldV1 = path.join(Truncate.DIR, v1Name(Date.now() - 10 * DAY_MS))
        const recentV1 = path.join(Truncate.DIR, v1Name(Date.now() - 3 * DAY_MS))

        yield* writeFileStringScoped(old, "old content")
        yield* writeFileStringScoped(recent, "recent content")
        yield* writeFileStringScoped(oldV1, "old v1")
        yield* writeFileStringScoped(recentV1, "recent v1")
        yield* svc.cleanup()

        expect(yield* fs.exists(old)).toBe(false)
        expect(yield* fs.exists(recent)).toBe(true)
        expect(yield* fs.exists(oldV1)).toBe(false)
        expect(yield* fs.exists(recentV1)).toBe(true)
      }),
    )
  })
})

describe("Truncate working-set governor", () => {
  const BUDGET_ENV = "MIMOCODE_WORKING_SET_BUDGET_BYTES"
  // Flag reads process.env per access, so pinning the budget here is enough.
  const withBudget = <A, E, R>(bytes: number, effect: Effect.Effect<A, E, R>) =>
    Effect.acquireUseRelease(
      Effect.sync(() => {
        const prev = process.env[BUDGET_ENV]
        process.env[BUDGET_ENV] = String(bytes)
        return prev
      }),
      () => effect,
      (prev) =>
        Effect.sync(() => {
          if (prev === undefined) delete process.env[BUDGET_ENV]
          else process.env[BUDGET_ENV] = prev
        }),
    )

  it.live("bounds the AGGREGATE inline output for a session without mutating earlier results", () =>
    withBudget(
      100 * 1024,
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const sessionID = "ses_working_set_test"
        const chunk = "x".repeat(40 * 1024)

        const first = yield* svc.output(chunk, {}, undefined, sessionID)
        const second = yield* svc.output(chunk, {}, undefined, sessionID)
        const third = yield* svc.output(chunk, {}, undefined, sessionID)
        const fourth = yield* svc.output(chunk, {}, undefined, sessionID)

        // 40 KiB, then 80 KiB of a 100 KiB budget → both fit whole.
        expect(first.truncated).toBe(false)
        expect(second.truncated).toBe(false)
        // Only ~20 KiB left → the third is capped to what remains.
        expect(third.truncated).toBe(true)
        // Budget spent → stub cap; still spilled and reachable via Read.
        expect(fourth.truncated).toBe(true)
        if (fourth.truncated) expect(fourth.outputPath).toBeDefined()

        // Earlier results are untouched: the bound was applied at insertion.
        expect(first.content).toBe(chunk)
        expect(second.content).toBe(chunk)
        expect(Buffer.byteLength(fourth.content, "utf-8")).toBeLessThan(Buffer.byteLength(third.content, "utf-8"))
      }),
    ),
  )

  it.live("reset clears the accounting so a rebuilt session starts over", () =>
    withBudget(
      100 * 1024,
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const sessionID = "ses_working_set_reset_test"
        const chunk = "y".repeat(40 * 1024)

        yield* svc.output(chunk, {}, undefined, sessionID)
        yield* svc.output(chunk, {}, undefined, sessionID)
        const spent = yield* svc.output(chunk, {}, undefined, sessionID)
        expect(spent.truncated).toBe(true)

        yield* svc.reset(sessionID)
        const afterReset = yield* svc.output(chunk, {}, undefined, sessionID)
        expect(afterReset.truncated).toBe(false)
      }),
    ),
  )

  it.live("without a sessionID the legacy per-result cap applies (no aggregate bound)", () =>
    Effect.gen(function* () {
      const svc = yield* Truncate.Service
      const chunk = "z".repeat(40 * 1024)
      const a = yield* svc.output(chunk)
      const b = yield* svc.output(chunk)
      expect(a.truncated).toBe(false)
      expect(b.truncated).toBe(false)
    }),
  )

  it.live("accounts the budget per ACTOR SLICE, not per session", () =>
    withBudget(
      100 * 1024,
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const sessionID = "ses_slice_test"
        const chunk = "s".repeat(40 * 1024)

        // Spend 80 KiB of the main slice's 100 KiB budget.
        yield* svc.output(chunk, {}, undefined, sessionID, "main")
        yield* svc.output(chunk, {}, undefined, sessionID, "main")
        const mainThird = yield* svc.output(chunk, {}, undefined, sessionID, "main")
        expect(mainThird.truncated).toBe(true)

        // A subagent slice on the SAME session keeps its own full budget: its
        // tool output never enters the parent's context, so it must not consume
        // the parent's budget.
        const subFirst = yield* svc.output(chunk, {}, undefined, sessionID, "explore-1")
        expect(subFirst.truncated).toBe(false)
      }),
    ),
  )

  it.live("reset clears every actor slice for the session", () =>
    withBudget(
      100 * 1024,
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const sessionID = "ses_slice_reset_test"
        const chunk = "r".repeat(40 * 1024)

        yield* svc.output(chunk, {}, undefined, sessionID, "main")
        yield* svc.output(chunk, {}, undefined, sessionID, "main")
        yield* svc.output(chunk, {}, undefined, sessionID, "explore-1")
        yield* svc.output(chunk, {}, undefined, sessionID, "explore-1")

        yield* svc.reset(sessionID)

        expect((yield* svc.output(chunk, {}, undefined, sessionID, "main")).truncated).toBe(false)
        expect((yield* svc.output(chunk, {}, undefined, sessionID, "explore-1")).truncated).toBe(false)
      }),
    ),
  )

  // `read` / `bash` / `grep` set metadata.truncated on EVERY result, so the tool
  // wrapper used to return early and the governor never saw them — i.e. it was
  // inert for exactly the tools whose output accumulates. A self-truncated
  // result must still be ACCOUNTED, and still be shrunk once the budget is spent.
  it.live("accounts a self-truncated result, then shrinks it once the budget is spent", () =>
    withBudget(
      100 * 1024,
      Effect.gen(function* () {
        const svc = yield* Truncate.Service
        const sessionID = "ses_self_trunc_test"
        const big = "x".repeat(60 * 1024)

        // Room left → a self-truncated result is passed through untouched.
        const first = yield* svc.output(big, { selfTruncated: true }, undefined, sessionID)
        expect(first.truncated).toBe(false)
        expect(first.content).toBe(big)

        // ...but it was ACCOUNTED (60 KiB of the 100 KiB budget), so the next
        // self-truncated result is shrunk by the governor.
        const second = yield* svc.output(big, { selfTruncated: true }, undefined, sessionID)
        expect(second.truncated).toBe(true)
        if (second.truncated) expect(second.outputPath).toBeDefined()
        expect(Buffer.byteLength(second.content, "utf-8")).toBeLessThan(Buffer.byteLength(big, "utf-8"))
      }),
    ),
  )
})
