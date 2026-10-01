import { createMemo, type Setter } from "solid-js"
import { useKV } from "./kv"

export type ThinkingMode = "show" | "hide"

const MODES: readonly ThinkingMode[] = ["show", "hide"] as const

export function reasoningSummary(text: string) {
  const content = text.trim()
  const match = content.match(/^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)/)
  if (!match) return { title: null, body: content }
  return { title: match[1].trim(), body: content.slice(match[0].length).trimEnd() }
}

/**
 * The header title on its own, without the body.
 *
 * `reasoningSummary` slices the body out of the whole text, and that slice is a real
 * allocation proportional to the stream. A caller that only wants the title must not
 * pay for it: the regex is anchored at `^`, so a text that does not START with `**`
 * cannot produce a title at all, and the capture `[^*\n]+` cannot cross a newline —
 * both facts are checked here in O(1) instead of O(text).
 */
export function reasoningTitle(text: string) {
  if (!text.startsWith("**")) return null
  const match = text.match(/^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)/)
  return match ? match[1].trim() : null
}

export function isThinkingMode(value: unknown): value is ThinkingMode {
  return typeof value === "string" && (MODES as readonly string[]).includes(value)
}

export function nextThinkingMode(current: ThinkingMode): ThinkingMode {
  const idx = MODES.indexOf(current)
  return MODES[(idx + 1) % MODES.length] ?? "show"
}

export function useThinkingMode() {
  const kv = useKV()
  const hadStored = kv.get("thinking_mode") !== undefined
  const legacy = kv.get("thinking_visibility")
  const [stored, setStored] = kv.signal<ThinkingMode>("thinking_mode", "hide")

  const set = (next: ThinkingMode | ((prev: ThinkingMode) => ThinkingMode)) => {
    if (typeof next === "function") setStored(next as Setter<ThinkingMode>)
    else setStored(() => next)
  }

  if (!hadStored) {
    if (legacy === true) set("show")
    else if (legacy === false) set("hide")
  }

  if ((stored() as string) === "minimal") set("hide")

  const mode = createMemo<ThinkingMode>(() => {
    const value = stored()
    return isThinkingMode(value) ? value : "hide"
  })

  return { mode, set }
}
