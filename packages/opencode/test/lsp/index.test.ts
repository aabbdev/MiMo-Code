import { describe, expect, spyOn } from "bun:test"
import path from "path"
import { Effect, Layer } from "effect"
import { LSP } from "../../src/lsp"
import { LSPServer } from "../../src/lsp"
import * as CrossSpawnSpawner from "../../src/effect/cross-spawn-spawner"
import { provideTmpdirInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"

const it = testEffect(Layer.mergeAll(LSP.defaultLayer, CrossSpawnSpawner.defaultLayer))

describe("lsp.spawn", () => {
  it.live("does not spawn builtin LSP for files outside instance", () =>
    provideTmpdirInstance(
      (dir) =>
        LSP.Service.use((lsp) =>
          Effect.gen(function* () {
            const spy = spyOn(LSPServer.Typescript, "spawn").mockResolvedValue(undefined)

            try {
              yield* lsp.touchFile(path.join(dir, "..", "outside.ts"))
              yield* lsp.hover({
                file: path.join(dir, "..", "hover.ts"),
                line: 0,
                character: 0,
              })
              expect(spy).toHaveBeenCalledTimes(0)
            } finally {
              spy.mockRestore()
            }
          }),
        ),
      { config: { lsp: true }, git: true },
    ),
  )

  it.live("does not spawn builtin LSP for files inside instance when LSP is unset", () =>
    provideTmpdirInstance((dir) =>
      LSP.Service.use((lsp) =>
        Effect.gen(function* () {
          const spy = spyOn(LSPServer.Typescript, "spawn").mockResolvedValue(undefined)

          try {
            yield* lsp.hover({
              file: path.join(dir, "src", "inside.ts"),
              line: 0,
              character: 0,
            })
            expect(spy).toHaveBeenCalledTimes(0)
          } finally {
            spy.mockRestore()
          }
        }),
      ),
    ),
  )

  it.live("would spawn builtin LSP for files inside instance when lsp is true", () =>
    provideTmpdirInstance(
      (dir) =>
        LSP.Service.use((lsp) =>
          Effect.gen(function* () {
            const spy = spyOn(LSPServer.Typescript, "spawn").mockResolvedValue(undefined)

            try {
              yield* lsp.hover({
                file: path.join(dir, "src", "inside.ts"),
                line: 0,
                character: 0,
              })
              expect(spy).toHaveBeenCalledTimes(1)
            } finally {
              spy.mockRestore()
            }
          }),
        ),
      { config: { lsp: true } },
    ),
  )

  it.live("would spawn builtin LSP for files inside instance when config object is provided", () =>
    provideTmpdirInstance(
      (dir) =>
        LSP.Service.use((lsp) =>
          Effect.gen(function* () {
            const spy = spyOn(LSPServer.Typescript, "spawn").mockResolvedValue(undefined)

            try {
              yield* lsp.hover({
                file: path.join(dir, "src", "inside.ts"),
                line: 0,
                character: 0,
              })
              expect(spy).toHaveBeenCalledTimes(1)
            } finally {
              spy.mockRestore()
            }
          }),
        ),
      {
        config: {
          lsp: {
            eslint: { disabled: true },
          },
        },
      },
    ),
  )
})

/**
 * MLIR is registered by export alone (`lsp.ts` enumerates `Object.values(LSPServer)`),
 * so these two assertions are what keep it from silently disappearing: one that
 * the entry exists with the right extensions, one that a `.mlir` file actually
 * reaches its spawn.
 */
describe("lsp.spawn mlir", () => {
  it.live("registers the MLIR server for .mlir and NOT for .td", () =>
    Effect.gen(function* () {
      expect(LSPServer.Mlir).toBeDefined()
      expect(LSPServer.Mlir.extensions).toContain(".mlir")
      // TableGen has no language server: an entry would spawn a client that can
      // never answer, which is worse than having none.
      expect(LSPServer.Mlir.extensions).not.toContain(".td")
    }),
  )

  it.live("would spawn the MLIR server for a .mlir file inside the instance", () =>
    provideTmpdirInstance(
      (dir) =>
        LSP.Service.use((lsp) =>
          Effect.gen(function* () {
            const spy = spyOn(LSPServer.Mlir, "spawn").mockResolvedValue(undefined)

            try {
              yield* lsp.hover({
                file: path.join(dir, "src", "dialect.mlir"),
                line: 0,
                character: 0,
              })
              expect(spy).toHaveBeenCalledTimes(1)
            } finally {
              spy.mockRestore()
            }
          }),
        ),
      { config: { lsp: true } },
    ),
  )
})
