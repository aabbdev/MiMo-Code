/** @jsxImportSource @opentui/solid */
import { expect, test } from "bun:test"
import { testRender } from "@opentui/solid"
import { ContextSidebar } from "../../../src/cli/cmd/tui/feature-plugins/sidebar/context"
import { createTuiPluginApi } from "../../fixture/tui-plugin"

test("sidebar compares the configured context limit with the model limit", async () => {
  const api = createTuiPluginApi({
    state: {
      config: { compaction: { max_context: { "test/gpt": 300_000 } } } as never,
      provider: [
        {
          id: "test",
          models: {
            gpt: {
              id: "gpt",
              providerID: "test",
              api: { id: "gpt", npm: "@ai-sdk/openai" },
              limit: { context: 922_000, input: 922_000, output: 128_000 },
            },
          },
        },
      ] as never,
      session: {
        messages: () =>
          [
            {
              id: "msg",
              role: "assistant",
              providerID: "test",
              modelID: "gpt",
              cost: 0,
              tokens: { input: 100_000, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
              time: { created: 1, completed: 2 },
            },
          ] as never,
      },
    },
  })

  const app = await testRender(() => <ContextSidebar api={api} session_id="ses" />)
  await app.renderOnce()

  expect(app.captureCharFrame()).toContain("33% used")
  expect(app.captureCharFrame()).toContain("limit 300K of 922K")
})

test("sidebar shows the session total, not a sum of the messages it holds", async () => {
  const api = createTuiPluginApi({
    state: {
      config: {} as never,
      provider: [] as never,
      session: {
        // The page the TUI holds sums to $1.00; the session really spent $17.30.
        // Summing the page is what made the readout plateau on a long session.
        messages: () =>
          [
            {
              id: "msg",
              role: "assistant",
              providerID: "test",
              modelID: "gpt",
              cost: 1,
              tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
              time: { created: 1, completed: 2 },
            },
          ] as never,
        cost: () => 17.3,
      },
    },
  })

  const app = await testRender(() => <ContextSidebar api={api} session_id="ses" />)
  await app.renderOnce()

  expect(app.captureCharFrame()).toContain("17.30")
  expect(app.captureCharFrame()).not.toContain("1.00")
})
