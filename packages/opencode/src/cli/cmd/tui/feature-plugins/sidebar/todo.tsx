import type { TuiPlugin, TuiPluginApi, TuiPluginModule } from "@mimo-ai/plugin/tui"
import { createMemo, For, Show, createSignal } from "solid-js"
import { TodoItem } from "../../component/todo-item"

const id = "internal:sidebar-todo"

/**
 * How many finished items the sidebar keeps. A long-lived session accumulates one
 * completed entry per finished task, and rendering the whole list turned the
 * sidebar into a log (83 rows on one dpu session). Every outstanding item is
 * always shown; finished ones only serve as recent progress, so a few suffice.
 */
const FINISHED_SHOWN = 3

function View(props: { api: TuiPluginApi; session_id: string }) {
  const [open, setOpen] = createSignal(true)
  const theme = () => props.api.theme.current
  const list = createMemo(() => props.api.state.session.todo(props.session_id))
  const tasks = createMemo(() => props.api.state.session.task(props.session_id))
  // `list()` is ordered by creation, so the tail holds the most recent finishes.
  const visible = createMemo(() => {
    const outstanding = list().filter((item) => item.status !== "completed" && item.status !== "cancelled")
    const finished = list().filter((item) => item.status === "completed" || item.status === "cancelled")
    return [...outstanding, ...finished.slice(-FINISHED_SHOWN)]
  })
  const hidden = createMemo(() => list().length - visible().length)
  const show = createMemo(
    () => tasks().length === 0 && list().length > 0 && list().some((item) => item.status !== "completed"),
  )

  return (
    <Show when={show()}>
      <box>
        <box flexDirection="row" gap={1} onMouseDown={() => visible().length > 2 && setOpen((x) => !x)}>
          <Show when={visible().length > 2}>
            <text fg={theme().text}>{open() ? "▼" : "▶"}</text>
          </Show>
          <text fg={theme().text}>
            <b>Todo</b>
          </text>
        </box>
        <Show when={visible().length <= 2 || open()}>
          <For each={visible()}>{(item) => <TodoItem status={item.status} content={item.content} />}</For>
          <Show when={hidden() > 0}>
            <text fg={theme().textMuted}>{hidden()} finished hidden</text>
          </Show>
        </Show>
      </box>
    </Show>
  )
}

const tui: TuiPlugin = async (api) => {
  api.slots.register({
    order: 400,
    slots: {
      sidebar_content(_ctx, props) {
        return <View api={api} session_id={props.session_id} />
      },
    },
  })
}

const plugin: TuiPluginModule & { id: string } = {
  id,
  tui,
}

export default plugin
