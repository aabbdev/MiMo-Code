import { describe, expect, test } from "bun:test"
import path from "path"
import fs from "node:fs/promises"
import { Instance } from "../../src/project/instance"
import { readDirectory, resolveInside } from "../../src/rlm/payload"
import { hitRate, tallyCost } from "../../src/tool/rlm"
import { tmpdir } from "../fixture/fixture"

/** Together's published prices for the model these runs used. */
const PRICE = { input: 0.3, output: 1.2, cacheRead: 0.006 }

describe("run cost accounting", () => {
  test("cached tokens bill at the cache rate, the rest at the input rate", () => {
    const cost = tallyCost({ input: 100_000, output: 10_000, cacheRead: 90_000 }, PRICE)
    expect(cost).toBeCloseTo((10_000 * 0.3 + 90_000 * 0.006 + 10_000 * 1.2) / 1e6, 12)
  })

  test("ignoring the cache split would overstate the cost by more than 10x", () => {
    const tally = { input: 100_000, output: 0, cacheRead: 95_000 }
    const naive = (tally.input * PRICE.input) / 1e6
    expect(naive / tallyCost(tally, PRICE)).toBeGreaterThan(10)
  })

  test("hit rate is measured against total input", () => {
    expect(hitRate({ input: 1000, output: 0, cacheRead: 250 })).toBeCloseTo(0.25, 12)
    expect(hitRate({ input: 0, output: 0, cacheRead: 0 })).toBe(0)
    // A measured run: the root prefix cached well while the sub-calls could not,
    // so the two rates must be read apart, never as one average.
    expect(hitRate({ input: 84_803, output: 0, cacheRead: 75_900 })).toBeGreaterThan(0.89)
    expect(hitRate({ input: 575_170, output: 0, cacheRead: 0 })).toBe(0)
  })

  test("a mid-stream report with cacheRead above input cannot yield a negative cost", () => {
    expect(tallyCost({ input: 100, output: 0, cacheRead: 200 }, PRICE)).toBeGreaterThan(0)
  })
})

async function inside<T>(dir: string, fn: () => Promise<T>): Promise<T> {
  return Instance.provide({ directory: dir, fn })
}

describe("rlm payload loading", () => {
  test("a file inside the worktree is read whole", async () => {
    await using tmp = await tmpdir({ git: true })
    await fs.mkdir(path.join(tmp.path, "src"), { recursive: true })
    await fs.writeFile(path.join(tmp.path, "src", "big.txt"), "line one\nline two\n")
    const text = await inside(tmp.path, async () => readDirectory(await resolveInside("src/big.txt")))
    expect(text.text).toBe("line one\nline two\n")
    expect(text.files).toBe(1)
  })

  test("a directory is concatenated with per-file headers so the model can address parts", async () => {
    await using tmp = await tmpdir({ git: true })
    await fs.mkdir(path.join(tmp.path, "mod"), { recursive: true })
    await fs.writeFile(path.join(tmp.path, "mod", "a.mlir"), "func.func @a()\n")
    await fs.writeFile(path.join(tmp.path, "mod", "b.mlir"), "func.func @b()\n")
    const out = await inside(tmp.path, async () => readDirectory(await resolveInside("mod")))
    expect(out.files).toBe(2)
    expect(out.text).toContain("### ")
    expect(out.text).toContain("func.func @a()")
    expect(out.text).toContain("func.func @b()")
  })

  test("binary and vendored paths are skipped rather than poisoning the payload", async () => {
    await using tmp = await tmpdir({ git: true })
    await fs.mkdir(path.join(tmp.path, "node_modules", "dep"), { recursive: true })
    await fs.writeFile(path.join(tmp.path, "node_modules", "dep", "index.js"), "module.exports = 1\n")
    await fs.writeFile(path.join(tmp.path, "logo.png"), "not really a png")
    await fs.writeFile(path.join(tmp.path, "keep.c"), "int main() { return 0; }\n")
    const out = await inside(tmp.path, async () => readDirectory(await resolveInside(".")))
    expect(out.text).toContain("keep.c")
    expect(out.text).not.toContain("module.exports")
    expect(out.text).not.toContain("not really a png")
  })

  test("a path outside the worktree is refused, including via a traversal or a symlink", async () => {
    await using tmp = await tmpdir({ git: true })
    await inside(tmp.path, async () => {
      // /etc/passwd exists on every Linux host; the point is the jail, not the file.
      await expect(resolveInside("/etc/passwd")).rejects.toThrow(/outside the worktree/)
      // Enough `..` to clear the fixture's own depth (it nests under the package
      // dir) so this exercises the jail rather than landing on a missing path.
      await expect(resolveInside("../".repeat(12) + "etc/passwd")).rejects.toThrow(/outside the worktree/)
      await expect(resolveInside("does/not/exist")).rejects.toThrow(/no such file or directory/)
    })
    // A symlink that points out of the tree must not become a door: the check
    // runs on the REALPATH, so the link resolves to its target and is refused.
    await fs.symlink("/etc/passwd", path.join(tmp.path, "escape"))
    await expect(inside(tmp.path, () => resolveInside("escape"))).rejects.toThrow(/outside the worktree/)
  })

  test("secrets are never swept into a directory payload, though an explicit path still reads one", async () => {
    await using tmp = await tmpdir({ git: true })
    await fs.writeFile(path.join(tmp.path, ".env"), "API_KEY=sk-live-do-not-ship\n")
    await fs.mkdir(path.join(tmp.path, ".ssh"), { recursive: true })
    await fs.writeFile(path.join(tmp.path, ".ssh", "id_rsa"), "PRIVATE-KEY-MATERIAL\n")
    await fs.writeFile(path.join(tmp.path, "server.pem"), "PRIVATE-KEY-MATERIAL\n")
    await fs.writeFile(path.join(tmp.path, "credentials.json"), '{"token":"do-not-ship"}\n')
    await fs.writeFile(path.join(tmp.path, "keep.mlir"), "func.func @k() {\n  return\n}\n")

    const out = await inside(tmp.path, async () => readDirectory(await resolveInside(".")))
    expect(out.text).toContain("keep.mlir")
    // A directory payload is handed to sub-models, so an IMPLICIT collection
    // must not carry credentials off the machine.
    expect(out.text).not.toContain("do-not-ship")
    expect(out.text).not.toContain("PRIVATE-KEY-MATERIAL")

    // An EXPLICIT path is the agent's own decision — the same trust its read
    // tool carries — so it is not filtered.
    const explicit = await inside(tmp.path, async () => readDirectory(await resolveInside(".env")))
    expect(explicit.text).toContain("do-not-ship")
  })

  test("the payload is exposed as parts, so the loop has a unit to fan out over", async () => {
    await using tmp = await tmpdir({ git: true })
    await fs.mkdir(path.join(tmp.path, "mod"), { recursive: true })
    await fs.writeFile(path.join(tmp.path, "mod", "a.mlir"), "func.func @a()\n")
    await fs.writeFile(path.join(tmp.path, "mod", "b.mlir"), "func.func @b()\n")
    const out = await inside(tmp.path, async () => readDirectory(await resolveInside("mod")))
    expect(out.names).toEqual(["a.mlir", "b.mlir"])
    expect(out.parts).toEqual(["func.func @a()\n", "func.func @b()\n"])
    // `context` stays the joined view, with headers, for whole-payload questions.
    expect(out.text).toContain("### a.mlir")
    expect(out.text).toContain("### b.mlir")
  })

  test("a file too large for one sub-call is split at line boundaries, never mid-statement", async () => {
    await using tmp = await tmpdir({ git: true })
    const lines = Array.from({ length: 4000 }, (_, i) => `int fn_${i}(void) { return ${i}; }`)
    await fs.writeFile(path.join(tmp.path, "big.cpp"), lines.join("\n"))
    const out = await inside(tmp.path, async () => readDirectory(await resolveInside("big.cpp")))
    expect(out.parts.length).toBeGreaterThan(1)
    expect(out.names[1]).toBe("big.cpp#2")
    // Every part stays inside the per-sub-call budget...
    expect(Math.max(...out.parts.map((p) => p.length))).toBeLessThanOrEqual(48_100)
    // ...and splitting loses nothing: joined back, every line is still there.
    expect(out.parts.join("").replace(/\n/g, "")).toBe(lines.join("").replace(/\n/g, ""))
  })

  test("a directory with no readable text is an error, not a silent empty payload", async () => {
    await using tmp = await tmpdir({ git: true })
    await fs.mkdir(path.join(tmp.path, "empty"), { recursive: true })
    await expect(inside(tmp.path, async () => readDirectory(await resolveInside("empty")))).rejects.toThrow(
      /no readable text files/,
    )
  })
})
