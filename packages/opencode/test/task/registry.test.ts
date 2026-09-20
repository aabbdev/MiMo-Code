import { afterEach, describe, expect } from "bun:test"
import { Cause, Effect, Layer } from "effect"
import { Bus } from "../../src/bus"
import { Session } from "../../src/session"
import { TaskRegistry } from "../../src/task/registry"
import { Instance } from "../../src/project/instance"
import { provideTmpdirInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { isRecoverableError } from "../../src/tool/recoverable"
import { Database, and, eq } from "../../src/storage"
import { TaskTable } from "../../src/task/task.sql"
import * as CrossSpawnSpawner from "../../src/effect/cross-spawn-spawner"

afterEach(async () => {
  await Instance.disposeAll()
})

const env = Layer.mergeAll(
  CrossSpawnSpawner.defaultLayer,
  Bus.defaultLayer,
  Session.defaultLayer,
  TaskRegistry.defaultLayer,
)

const it = testEffect(env)

const seedSession = Effect.fn("Test.seedSession")(function* () {
  const session = yield* Session.Service
  return yield* session.create({ title: "Test" })
})

describe("TaskRegistry.create", () => {
  it.live("creates a top-level task with id T1", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()

        const task = yield* reg.create({
          session_id: sess.id,
          summary: "Refactor auth",
        })
        expect(task.id).toBe("T1")
        expect(task.status).toBe("open")
        expect(task.parent_task_id).toBeUndefined()
      }),
    ),
  )

  it.live("creates sequential top-level ids T1, T2, T3", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()

        const t1 = yield* reg.create({ session_id: sess.id, summary: "a" })
        const t2 = yield* reg.create({ session_id: sess.id, summary: "b" })
        const t3 = yield* reg.create({ session_id: sess.id, summary: "c" })
        expect(t1.id).toBe("T1")
        expect(t2.id).toBe("T2")
        expect(t3.id).toBe("T3")
      }),
    ),
  )

  it.live("creates subtask T1.1 under T1", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()

        const t1 = yield* reg.create({ session_id: sess.id, summary: "parent" })
        const sub = yield* reg.create({ session_id: sess.id, summary: "child", parent_id: t1.id })
        expect(sub.id).toBe("T1.1")
        expect(sub.parent_task_id).toBe("T1")
      }),
    ),
  )

  it.live("emits 'created' task_event on create", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()
        const t = yield* reg.create({ session_id: sess.id, summary: "x" })
        const events = yield* reg.events({ session_id: sess.id, task_id: t.id })
        expect(events.length).toBe(1)
        expect(events[0].kind).toBe("created")
      }),
    ),
  )

  it.live("two sessions can each have a T1 without colliding", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const a = yield* seedSession()
        const b = yield* seedSession()

        const ta = yield* reg.create({ session_id: a.id, summary: "in A" })
        const tb = yield* reg.create({ session_id: b.id, summary: "in B" })
        expect(ta.id).toBe("T1")
        expect(tb.id).toBe("T1")
        expect(ta.session_id).toBe(a.id)
        expect(tb.session_id).toBe(b.id)
      }),
    ),
  )
})

describe("TaskRegistry not-found is agent-recoverable", () => {
  it.live("start on a nonexistent id dies with an actionable RecoverableError", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()

        const exit = yield* Effect.exit(reg.start({ session_id: sess.id, id: "T99" }))
        expect(exit._tag).toBe("Failure")
        if (exit._tag !== "Failure") return
        const err = Cause.squash(exit.cause)
        expect(isRecoverableError(err)).toBe(true)
        expect((err as Error).message).toContain("task list")
      }),
    ),
  )
})

describe("TaskRegistry.list", () => {
  it.live("lists active tasks for a session by default", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()
        yield* reg.create({ session_id: sess.id, summary: "a" })
        yield* reg.create({ session_id: sess.id, summary: "b" })

        const list = yield* reg.list({ session_id: sess.id })
        expect(list.length).toBe(2)
      }),
    ),
  )

  it.live("excludes terminal tasks by default", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()
        const t1 = yield* reg.create({ session_id: sess.id, summary: "a" })
        yield* reg.done({ session_id: sess.id, id: t1.id })
        yield* reg.create({ session_id: sess.id, summary: "b" })

        const list = yield* reg.list({ session_id: sess.id })
        expect(list.length).toBe(1)
        expect(list[0].summary).toBe("b")
      }),
    ),
  )

  it.live("includes terminal when include_terminal=true", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()
        const t1 = yield* reg.create({ session_id: sess.id, summary: "a" })
        yield* reg.done({ session_id: sess.id, id: t1.id })

        const list = yield* reg.list({ session_id: sess.id, include_terminal: true })
        expect(list.length).toBe(1)
      }),
    ),
  )
})

describe("TaskRegistry.cleanup", () => {
  // `cleanup_after` is stamped at done()/abandon() time from config (7 days by
  // default); backdate it to simulate a window that has elapsed. Task ids are
  // per-session (the PK is session_id + id), so the session must be part of the
  // filter — keying on `id` alone reaches "T1" in every other session too.
  const elapseWindow = (session_id: string, id: string) =>
    Effect.sync(() =>
      Database.use((db) =>
        db
          .update(TaskTable)
          .set({ cleanup_after: Date.now() - 1000 })
          .where(and(eq(TaskTable.session_id, session_id as never), eq(TaskTable.id, id)))
          .run(),
      ),
    )

  it.live("removes terminal tasks past their archive window, with their events", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()

        const finished = yield* reg.create({ session_id: sess.id, summary: "finished long ago" })
        yield* reg.start({ session_id: sess.id, id: finished.id })
        yield* reg.done({ session_id: sess.id, id: finished.id, event_summary: "done" })
        expect((yield* reg.events({ session_id: sess.id, task_id: finished.id })).length).toBeGreaterThan(0)
        yield* elapseWindow(sess.id, finished.id)

        const live = yield* reg.create({ session_id: sess.id, summary: "still open" })

        expect(yield* reg.cleanup()).toBe(1)

        expect(yield* reg.get({ session_id: sess.id, id: finished.id })).toBeUndefined()
        // The composite FK cascades, so the events went with it.
        expect(yield* reg.events({ session_id: sess.id, task_id: finished.id })).toEqual([])
        // An actionable task is never touched, and the row is really gone — not
        // merely hidden: include_archived must not resurrect it.
        expect(yield* reg.get({ session_id: sess.id, id: live.id })).toBeDefined()
        const remaining = yield* reg.list({ session_id: sess.id, include_terminal: true, include_archived: true })
        expect(remaining.map((t) => t.id)).toEqual([live.id])
      }),
    ),
  )

  it.live("keeps terminal tasks whose archive window has not elapsed", () =>
    provideTmpdirInstance(() =>
      Effect.gen(function* () {
        const reg = yield* TaskRegistry.Service
        const sess = yield* seedSession()
        const t = yield* reg.create({ session_id: sess.id, summary: "just finished" })
        yield* reg.done({ session_id: sess.id, id: t.id })

        expect(yield* reg.cleanup()).toBe(0)
        expect(yield* reg.get({ session_id: sess.id, id: t.id })).toBeDefined()
      }),
    ),
  )
})
