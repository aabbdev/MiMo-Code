import { describe, expect, test } from "bun:test"
import { reasoningSummary, reasoningTitle } from "../../../src/cli/cmd/tui/context/thinking"

describe("reasoningTitle", () => {
  /**
   * The premise this function exists for: the header title must be obtainable WITHOUT
   * the body. `reasoningSummary` slices the body out of the whole text, and that slice
   * is an allocation proportional to the stream — computing it per reasoning delta for
   * a header is what made ReasoningPart quadratic. So the lock is EQUIVALENCE: the
   * cheap path has to agree with the expensive one on every shape.
   */
  const shapes = [
    "**Reading the parser**\n\nI start by looking at the lexer.",
    "**Title**",
    "**Title**\n\n",
    "**Title**\r\n\r\nbody after a CRLF",
    "no title at all, just thinking",
    "Let me look first.\n\n**Not a title**",
    "",
    "   ",
    "**Title with a * inside**\n\nbody",
    "**unterminated title\n\nbody",
  ]

  test("agrees with reasoningSummary on every shape", () => {
    for (const raw of shapes) {
      const text = raw.replace("[REDACTED]", "").trim()
      expect(reasoningTitle(text)).toBe(reasoningSummary(text).title)
    }
  })

  test("a text that does not START with ** can never produce a title", () => {
    // This is the whole reason the title does not need to look past the first two
    // characters: the regex is anchored at `^`, so a title further down can only be
    // found by `reasoningSummary`, which is equally required to return null.
    const text = "Let me look first.\n\n**Not a title**"
    expect(reasoningTitle(text)).toBeNull()
    expect(reasoningSummary(text).title).toBeNull()
  })

  test("reads a real title out of the leading bold run", () => {
    expect(reasoningTitle("**Reading the parser**\n\nbody")).toBe("Reading the parser")
  })

  test("stops at the first newline, so a long body cannot become a title", () => {
    // The capture `[^*\n]+` cannot cross a newline — which is what bounds this to the
    // first line rather than to the length of the stream.
    expect(reasoningTitle("**short**\n\n" + "x".repeat(200_000))).toBe("short")
  })
})
