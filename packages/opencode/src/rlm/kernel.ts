/**
 * A persistent QuickJS kernel, one per session — the substrate for Recursive
 * Language Models (arXiv 2512.24601).
 *
 * `workflow/sandbox.ts` builds a runtime, runs ONE script, and disposes
 * everything. That is right for a replayable workflow and wrong for a REPL:
 * Algorithm 1 of the RLM paper puts the payload into the environment as a
 * variable and *iterates* over it, so the context and the model's working
 * variables must outlive a single call. This module retains them.
 *
 * Three rules make a retained context safe. All are load-bearing:
 *
 *  1. HOOK HANDLES ARE KERNEL-OWNED. `vm.dispose()` hard-aborts the process —
 *     not a leak warning, a crash — while any GC object is still alive, so a
 *     handle created once at setup must NOT be disposed when a call returns.
 *     Every handle produced DURING a call (marshalled results, deferred
 *     promises) instead belongs to that call's arena and is disposed with it.
 *     The two are tracked separately for exactly this reason.
 *
 *  2. PERSISTENCE NEEDS THE GLOBALS REWRITE. Host hooks are async (`llm_query`
 *     is a network call), so the guest body must run inside an async function
 *     to be allowed to `await` — and a `const`/`let` declared inside a function
 *     is function-local, i.e. gone by the next iteration. Evaluating the body as
 *     a top-level script instead would persist its lexical declarations but
 *     forbid `await` (measured: `SyntaxError: expecting ';'`). So
 *     `declareGlobals()` rewrites top-level declarations into assignments on
 *     `globalThis`, keeping line numbers intact. A bare assignment already
 *     creates a global, so the natural REPL style works either way.
 *
 *  3. ALL PER-CALL STATE IS PER-KERNEL. Two sessions run concurrently, so a
 *     module-level "current call" would hand one session's log buffer and
 *     sub-model binding to another's guest. Everything mutable lives in the
 *     closure built by `build()`.
 */
import ts from "typescript"
import {
  newQuickJSWASMModuleFromVariant,
  type QuickJSContext,
  type QuickJSDeferredPromise,
  type QuickJSHandle,
} from "quickjs-emscripten-core"
import singlefileVariant from "@jitl/quickjs-singlefile-mjs-release-sync"
import { formatGuestError, marshalIn, type HostFn } from "../workflow/sandbox"

type KernelRuntime = QuickJSContext["runtime"]

/** Value returned by one iteration of guest code. */
export type KernelStep = {
  value?: unknown
  logs: string[]
  error?: string
}

/** Per-call bindings. `llmQuery`/`llmQueryBatched` are what the guest sees as
 * `llm_query`/`llm_query_batched`; the RLM loop supplies a fresh pair per run so
 * the sub-call budget resets at the right boundary. */
export type RunContext = {
  llmQuery: HostFn
  llmQueryBatched: HostFn
  /**
   * The agent's own tools, reachable from the guest as `tools.<name>(args)`.
   *
   * Optional because a kernel with no bridge is still useful — it is then a
   * sandbox over the payload and nothing more. When it IS bound, the tools it
   * dispatches to are the request's WRAPPED ones, so permission asks, plugin
   * hooks and truncation apply exactly as they do to a direct call: the bridge
   * adds a caller, not a second authorization path.
   */
  callTool?: HostFn
  /**
   * A nested RLM over a sub-context, reachable from the guest as
   * `rlm_query(text, question)`. Depth > 1 in the paper's terms: the trajectory
   * spawns a whole RLM loop rather than a single sub-call, for sub-tasks that
   * themselves need chunking, aggregation or several steps. Bound only when the
   * caller allows that depth, and it always charges the parent's budget.
   */
  rlmQuery?: HostFn
  /**
   * A cheap first pass over an index of the payload: `(question, index) => indices
   * worth reading`. Bound by the caller because it makes model calls, and it is
   * what lets a trajectory narrow a 462-part payload before paying to read it.
   */
  screen?: HostFn
  /** Guest COMPUTE budget. Governs the interrupt handler, and is deliberately
   * separate from the wall clock: a step that makes twenty 30-second sub-calls
   * parks for ten minutes while spending almost no guest compute, and charging
   * it compute time would kill honest work. */
  computeMs?: number
  /** Absolute cap on the call, parked time included — the backstop for a host
   * promise that never settles. Exceeding either budget DISPOSES the kernel: a
   * guest parked mid-await cannot be resumed coherently. */
  wallMs?: number
  interrupt?: () => boolean
}

export type Kernel = {
  run(code: string, ctx: RunContext): Promise<KernelStep>
  /** Inject a value as a guest global without evaluating guest code. */
  set(name: string, value: unknown): void
}

/** 256 MiB, not the sandbox's 64: the payload IS the workload here (the paper
 * runs corpora of millions of tokens), and 10 MiB of text becomes several times
 * that once sliced into arrays of lines. */
const MEMORY_LIMIT = 256 * 1024 * 1024
const DEFAULT_COMPUTE_MS = 10 * 60 * 1000
const DEFAULT_WALL_MS = 60 * 60 * 1000
const IDLE_DISPOSE_MS = 30 * 60 * 1000
const PUMP_FAST_MS = 1
const PUMP_SLOW_MS = 50
const PUMP_FAST_WINDOW = 50

/** Guest-side shims. The bare QuickJS realm has no console; the RLM prompt tells
 * the model to print by logging, so logging must exist and must reach the host. */
const PRELUDE = `
const __fmt = (v) => {
  if (typeof v === "string") return v;
  if (v instanceof Error) return v.stack || String(v);
  try { const s = JSON.stringify(v); return s === undefined ? String(v) : s; } catch { return String(v); }
};
globalThis.console = {
  log: (...a) => __log(a.map(__fmt).join(" ")),
  error: (...a) => __log("[error] " + a.map(__fmt).join(" ")),
  warn: (...a) => __log("[warn] " + a.map(__fmt).join(" ")),
  info: (...a) => __log(a.map(__fmt).join(" ")),
};
// The public names the RLM prompt teaches. They are thin wrappers so the host
// can rebind the sub-model per run without re-injecting guest globals.
// A host rejection crosses the sandbox as a STRING, so the natural
// catch (e) { e.message } would be undefined and the model would lose every
// refusal reason the host took care to write. Normalised once, here, rather than
// in every snippet the model writes.
const __asError = (cause) =>
  cause instanceof Error ? cause : new Error(typeof cause === "string" ? cause : JSON.stringify(cause));
globalThis.llm_query = async (prompt) => {
  try { return await __llmQuery(prompt); } catch (e) { throw __asError(e); }
};
globalThis.llm_query_batched = async (prompts) => {
  try { return await __llmQueryBatched(prompts); } catch (e) { throw __asError(e); }
};
// A sub-call shaped for cache reuse: the CHUNK first, the QUESTION last.
//
// Every sub-call carries a distinct chunk, so it can never hit the provider's
// prefix cache -- except when the same chunk is asked about twice, which is the
// normal case for follow-up questions. llm_query sends whatever the caller writes
// first, which is usually the instructions, so nothing is reusable; this helper
// puts the payload first and the question after it, and the second question about
// a part then reads that part from cache instead of paying for it again.
// Which parts are worth reading, judged from their previews alone -- one cheap
// pass over a compact index instead of paying to read the whole payload.
// Returns the indices to read, so the trajectory then reads only those.
globalThis.screen = async (question, previewChars) => {
  try {
    if (!Array.isArray(globalThis.context_parts)) throw new Error("no payload is loaded: call load first");
    const size = typeof previewChars === "number" && previewChars > 0 ? previewChars : 400;
    const index = context_parts.map((part, i) => ({
      name: context_part_names[i],
      preview: String(part).slice(0, size),
    }));
    return await __screen(String(question), index);
  } catch (e) { throw __asError(e); }
};
// A whole nested RLM over a sub-context, for a sub-task too hard for one call.
globalThis.rlm_query = async (context, question) => {
  try { return await __rlmQuery(String(context), String(question)); } catch (e) { throw __asError(e); }
};
globalThis.ask_about = (index, question) => {
  if (!Array.isArray(globalThis.context_parts)) throw new Error("no payload is loaded: call load first");
  const part = context_parts[index];
  if (typeof part !== "string") throw new Error("no such part: " + index);
  // Doubled because this file is itself a template literal: a single backslash-n
  // would become a real newline here and leave the guest's string unterminated.
  return llm_query(part + "\\n\\n" + question);
};
// The agent's own tools, callable from inside the kernel:
//   const file = await tools.read({ file_path: "src/x.cpp" });
// A dynamic proxy rather than generated stubs: the surface is whatever the host
// authorizes for THIS request, so there is nothing to keep in sync, and an
// unauthorized name fails at the host with the real permission pipeline behind it.
// Symbol keys are excluded so awaiting the proxy or spreading it cannot be
// mistaken for a tool call.
// The context global as a LAZY join of the parts. Setting both meant the guest
// held the payload twice -- measured at +92 MB of process RSS for a 4.5 MB payload
// -- and a trajectory that only slices context_parts never paid for the copy.
Object.defineProperty(globalThis, "context", {
  configurable: true,
  get() {
    if (!Array.isArray(globalThis.context_parts)) throw new Error("no payload is loaded: call load first");
    if (globalThis.__joined === undefined) globalThis.__joined = context_parts.join("\\n\\n");
    return globalThis.__joined;
  },
});
globalThis.tools = new Proxy({}, {
  get: (_target, name) =>
    typeof name === "symbol"
      ? undefined
      : async (args) => {
          try { return await __callTool(String(name), args ?? {}); } catch (e) { throw __asError(e); }
        },
  has: () => true,
});
`

type CallScope = { arena: QuickJSHandle[]; deferreds: QuickJSDeferredPromise[]; logs: string[] }
type CallState = {
  wallDeadline: number
  /** Guest-compute budget, in ms. */
  activeBudget: number
  pending: number
  activeStart: number
  activeAccum: number
  interrupt?: () => boolean
}

const EMPTY_SCOPE: CallScope = { arena: [], deferreds: [], logs: [] }
const noScope = () => EMPTY_SCOPE

/**
 * Rewrite top-level declarations so their bindings land on `globalThis` and
 * survive the call, without moving any code to a different line (so transpile
 * diagnostics keep reporting the caller's own line numbers).
 *
 * Declarations bound through a destructuring pattern are left untouched: they
 * stay call-local rather than risk a rewrite whose evaluation order differs.
 */
export function declareGlobals(body: string): string {
  const sf = ts.createSourceFile("rlm.js", body, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS)
  const edits: { start: number; end: number; text: string }[] = []
  for (const stmt of sf.statements) {
    if (ts.isVariableStatement(stmt)) {
      const decls = stmt.declarationList.declarations
      if (!decls.every((d) => ts.isIdentifier(d.name))) continue
      // Delete through to the first binding, so the space that followed the
      // keyword goes with it — `const n = 1` must yield `globalThis.n = 1`, not
      // ` globalThis.n = 1`.
      edits.push({ start: stmt.declarationList.getStart(sf), end: decls[0]!.getStart(sf), text: "" })
      for (const d of decls) {
        const nameStart = d.name.getStart(sf)
        edits.push({ start: nameStart, end: nameStart, text: "globalThis." })
        // `x: number = 1` would otherwise become the invalid `globalThis.x: number = 1`.
        if (d.type) edits.push({ start: d.name.getEnd(), end: d.type.getEnd(), text: "" })
      }
      continue
    }
    if ((ts.isFunctionDeclaration(stmt) || ts.isClassDeclaration(stmt)) && stmt.name) {
      const start = stmt.getStart(sf)
      edits.push({ start, end: start, text: `globalThis.${stmt.name.text} = ` })
      edits.push({ start: stmt.getEnd(), end: stmt.getEnd(), text: ";" })
    }
  }
  return edits
    .sort((a, b) => b.start - a.start || b.end - a.end)
    .reduce((text, e) => text.slice(0, e.start) + e.text + text.slice(e.end), body)
}

/**
 * Inject host functions that outlive the call. The promise path mirrors
 * `sandbox.ts`: a host promise settles the guest promise and immediately drains
 * the guest's job queue, which is what makes `await` work without asyncify.
 */
/**
 * Give a bare trailing expression the value a console would show. The body runs
 * as an async function, so `context.length` on its own would evaluate and be
 * thrown away — a REPL that silently answers `undefined` teaches the model
 * nothing. Applied AFTER `declareGlobals`, so a rewritten declaration
 * (`const x = 1` → `globalThis.x = 1`) reports its value too.
 */
export function autoReturn(body: string): string {
  const sf = ts.createSourceFile("rlm.js", body, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS)
  const last = sf.statements[sf.statements.length - 1]
  if (!last || !ts.isExpressionStatement(last)) return body
  // Wrap the expression node, not the statement: the statement's range includes
  // its semicolon, and `return (x = 1;)` is a syntax error.
  const expr = last.expression
  const start = expr.getStart(sf)
  return body.slice(0, start) + "return (" + body.slice(start, expr.getEnd()) + ")" + body.slice(expr.getEnd())
}

function injectHooks(
  vm: QuickJSContext,
  hooks: Record<string, HostFn>,
  scope: () => CallScope,
  tracker: { start: () => void; end: () => void },
  owned: QuickJSHandle[],
): void {
  for (const [name, fn] of Object.entries(hooks)) {
    const handle = vm.newFunction(name, (...argHandles) => {
      const args = argHandles.map((h) => vm.dump(h))
      const out = fn(...args)
      if (out instanceof Promise) {
        const call = scope()
        const deferred = vm.newPromise()
        call.deferreds.push(deferred)
        tracker.start()
        out.then(
          (value) => {
            tracker.end()
            // A late settle can arrive after the realm was disposed (a step that
            // timed out). Bail before touching a dead context.
            if (!vm.alive) return
            const vh = marshalIn(vm, value)
            deferred.resolve(vh)
            vh.dispose()
            vm.runtime.executePendingJobs()
          },
          (err) => {
            tracker.end()
            if (!vm.alive) return
            const eh = vm.newString(err instanceof Error ? err.message : String(err))
            deferred.reject(eh)
            eh.dispose()
            vm.runtime.executePendingJobs()
          },
        )
        deferred.settled.then(() => {
          if (vm.alive) vm.runtime.executePendingJobs()
        })
        return deferred.handle
      }
      return marshalIn(vm, out)
    })
    vm.setProp(vm.global, name, handle)
    // Owned by the global object until the realm is torn down — see rule 1.
    owned.push(handle)
  }
}

function describeDiagnostics(code: string, diagnostics: readonly ts.Diagnostic[]): string {
  const lines = code.split("\n")
  const rendered = diagnostics
    .map((diagnostic) => {
      if (!diagnostic.file || diagnostic.start === undefined)
        return ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")
      // The wrapper adds exactly one line above the caller's first line, so a
      // 0-based position in the wrapped text is a 1-based line in the caller's.
      const pos = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start)
      return `line ${pos.line}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")}\n  ${lines[pos.line - 1] ?? ""}`
    })
    .join("\n")
  return `Transpile failed:\n${rendered}`
}

type Entry = {
  kernel: Kernel
  vm: QuickJSContext
  rt: KernelRuntime
  owned: QuickJSHandle[]
  used: number
}

async function build(sessionID: string): Promise<Entry> {
  modulePromise ??= newQuickJSWASMModuleFromVariant(singlefileVariant)
  const QuickJS = await modulePromise
  const rt = QuickJS.newRuntime()
  rt.setMemoryLimit(MEMORY_LIMIT)
  const vm = rt.newContext()

  // Per-kernel call state — see rule 3. `call` is what the hooks read; `state` is
  // what the interrupt handler reads; `host` is the sub-model binding.
  let call: CallScope | undefined
  let state: CallState | undefined
  let host: {
    llmQuery: HostFn
    llmQueryBatched: HostFn
    callTool?: HostFn
    rlmQuery?: HostFn
    screen?: HostFn
  } | undefined
  const tracker = {
    start: () => {
      const s = state
      if (!s) return
      if (s.pending === 0) s.activeAccum += Date.now() - s.activeStart
      s.pending++
    },
    end: () => {
      const s = state
      if (!s) return
      s.pending--
      if (s.pending === 0) s.activeStart = Date.now()
    },
  }

  rt.setInterruptHandler(() => {
    const s = state
    if (!s) return false
    if (s.interrupt?.()) return true
    // A call parked on a host hook is charged only for the wall clock (enforced
    // by the deadline race below), never for guest compute it is not spending.
    const elapsed = s.activeAccum + (s.pending === 0 ? Date.now() - s.activeStart : 0)
    if (elapsed > s.activeBudget) return true
    return Date.now() > s.wallDeadline && s.pending === 0
  })

  const owned: QuickJSHandle[] = []
  const prelude = vm.evalCode(PRELUDE)
  if (prelude.error) {
    const dumped = vm.dump(prelude.error)
    prelude.error.dispose()
    vm.dispose()
    rt.dispose()
    throw new Error(`RLM prelude failed: ${formatGuestError(dumped)}`)
  }
  prelude.value.dispose()

  injectHooks(
    vm,
    {
      __log: (line) => {
        call?.logs.push(String(line))
      },
      __llmQuery: (prompt) => host?.llmQuery(prompt) ?? Promise.reject(new Error("no sub-model bound")),
      __llmQueryBatched: (prompts) => host?.llmQueryBatched(prompts) ?? Promise.reject(new Error("no sub-model bound")),
      __callTool: (name, args) =>
        host?.callTool ? host.callTool(name, args) : Promise.reject(new Error("this kernel has no tool bridge")),
      __rlmQuery: (text, question) =>
        host?.rlmQuery ? host.rlmQuery(text, question) : Promise.reject(new Error("rlm_query is disabled at this depth")),
      __screen: (question, index) =>
        host?.screen ? host.screen(question, index) : Promise.reject(new Error("screening is not available here")),
    },
    () => call ?? EMPTY_SCOPE,
    tracker,
    owned,
  )

  const kernel: Kernel = {
    set(name, value) {
      const handle = marshalIn(vm, value)
      vm.setProp(vm.global, name, handle)
      handle.dispose()
    },
    async run(rawCode, ctx) {
      // Keep the realm alive while it is in use. `acquire` refreshes this too, but
      // a session-mode realm is acquired once and then run for as long as the
      // session lasts, so without this the idle sweeper would drop a kernel that is
      // working.
      const live = entries.get(sessionID)
      if (live) live.used = Date.now()
      // A realm that died on an unrecoverable step is gone; the caller's next
      // acquire rebuilds it. Say so instead of touching a dead context.
      if (!vm.alive) return { logs: [], error: "RLM kernel was reset — retry the call" }
      const scope: CallScope = { arena: [], deferreds: [], logs: [] }
      call = scope
      host = {
        llmQuery: ctx.llmQuery,
        llmQueryBatched: ctx.llmQueryBatched,
        callTool: ctx.callTool,
        rlmQuery: ctx.rlmQuery,
        screen: ctx.screen,
      }
      state = {
        wallDeadline: Date.now() + (ctx.wallMs ?? DEFAULT_WALL_MS),
        activeBudget: ctx.computeMs ?? DEFAULT_COMPUTE_MS,
        pending: 0,
        activeStart: Date.now(),
        activeAccum: 0,
        interrupt: ctx.interrupt,
      }
      let unrecoverable = false
      try {
        const wrapped = `globalThis.__step = async () => {\n${autoReturn(declareGlobals(rawCode))}\n}`
        const compiled = ts.transpileModule(wrapped, {
          reportDiagnostics: true,
          compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ESNext },
        })
        if (compiled.diagnostics?.length) return { logs: scope.logs, error: describeDiagnostics(rawCode, compiled.diagnostics) }
        const defined = vm.evalCode(compiled.outputText)
        if (defined.error) {
          const dumped = vm.dump(defined.error)
          defined.error.dispose()
          return { logs: scope.logs, error: formatGuestError(dumped) }
        }
        defined.value.dispose()
        return await drive(vm, rt, vm.evalCode("__step()"), scope, state, () => {
          unrecoverable = true
        })
      } finally {
        call = undefined
        host = undefined
        state = undefined
        // Dispose the CALL's handles only — never the kernel's hook handles.
        for (const d of scope.deferreds) if (d.alive) d.dispose()
        for (const h of scope.arena) if (h.alive) h.dispose()
        // ORDER MATTERS: the realm may only be dropped once every handle this
        // call created is gone — `vm.dispose()` aborts the process while any GC
        // object is still alive, which is a crash, not a catchable error.
        if (unrecoverable) disposeSession(sessionID)
      }
    },
  }

  return { kernel, vm, rt, owned, used: Date.now() }
}

/**
 * Settle the guest promise and race a wall deadline as the true kill-switch
 * (the runtime interrupt handler only fires while the guest executes bytecode,
 * so it cannot stop a guest parked on a host promise that never settles).
 */
async function drive(
  vm: QuickJSContext,
  rt: KernelRuntime,
  evaluated: ReturnType<QuickJSContext["evalCode"]>,
  scope: CallScope,
  state: CallState,
  markUnrecoverable: () => void,
): Promise<KernelStep> {
  if (evaluated.error) {
    const dumped = vm.dump(evaluated.error)
    evaluated.error.dispose()
    return { logs: scope.logs, error: formatGuestError(dumped) }
  }
  const promise = evaluated.value
  const settled = vm.resolvePromise(promise)
  // Adaptive pump: busy right after finding work, backed off once the guest has
  // been idle — an iteration parked on a slow llm_query would otherwise burn a
  // wakeup per millisecond for minutes. It never stops, so it cannot deadlock.
  let idle = 0
  let timer: ReturnType<typeof setTimeout> | undefined
  const pump = () => {
    if (rt.hasPendingJob()) {
      rt.executePendingJobs()
      idle = 0
    } else idle++
    timer = setTimeout(pump, idle < PUMP_FAST_WINDOW ? PUMP_FAST_MS : PUMP_SLOW_MS)
  }
  timer = setTimeout(pump, PUMP_FAST_MS)
  let deadline: ReturnType<typeof setTimeout> | undefined
  const expired = new Promise<never>((_, reject) => {
    deadline = setTimeout(
      () => reject(new Error("RLM step wall-clock limit exceeded")),
      Math.max(1, state.wallDeadline - Date.now()),
    )
  })
  try {
    const resolved = await Promise.race([settled, expired])
    if (resolved.error) {
      const dumped = vm.dump(resolved.error)
      resolved.error.dispose()
      return { logs: scope.logs, error: formatGuestError(dumped) }
    }
    const value = vm.dump(resolved.value)
    resolved.value.dispose()
    return { value, logs: scope.logs }
  } catch (err) {
    // The guest is parked mid-await with unsettled deferreds, so this realm can
    // never be resumed coherently — flag it for the caller, which drops THIS
    // session's realm only (never the other sessions') once the call's handles
    // are disposed. Disposing here would abort the process.
    markUnrecoverable()
    return { logs: scope.logs, error: err instanceof Error ? err.message : String(err) }
  } finally {
    clearTimeout(timer)
    clearTimeout(deadline)
    if (promise.alive) promise.dispose()
  }
}

// ---------------------------------------------------------------------------
// Registry — one realm per session, keyed exactly as the RLM tool addresses it.
// ---------------------------------------------------------------------------

let modulePromise: Promise<Awaited<ReturnType<typeof newQuickJSWASMModuleFromVariant>>> | undefined
let sweep: ReturnType<typeof setInterval> | undefined
const entries = new Map<string, Entry>()

/** Build (or reuse) the realm for `sessionID`. */
export async function acquire(sessionID: string): Promise<Kernel> {
  const existing = entries.get(sessionID)
  if (existing) {
    existing.used = Date.now()
    return existing.kernel
  }
  startSweeper()
  const entry = await build(sessionID)
  entries.set(sessionID, entry)
  return entry.kernel
}

/** Drop one session's realm. Safe to call when none exists, and safe to call
 * from inside a running step (the driver's unrecoverable path). */
export function disposeSession(sessionID: string): void {
  const entry = entries.get(sessionID)
  if (!entry) return
  entries.delete(sessionID)
  try {
    for (const handle of entry.owned) if (handle.alive) handle.dispose()
    entry.vm.dispose()
    entry.rt.dispose()
  } catch {
    // A realm already in a bad state must not take the caller down with it.
  }
}

/**
 * Drop everything belonging to one session.
 *
 * Realm keys are `<sessionID>` plus an optional suffix from a reserved set:
 * `#c<k>` for a candidate realm and `@repl` for the session's own kernel. Matching
 * on that convention rather than on a prefix is deliberate — `@` and `#` are not
 * prefixes of each other, so a `${sessionID}#` test would quietly leave the session
 * kernel alive, holding the payload it was supposed to release.
 *
 * This is the release a persistent kernel needs: an idle sweep alone leaves a
 * finished session's payload in memory for up to half an hour.
 */
export function disposeMatching(sessionID: string): number {
  const owner = (key: string) => key.split(/[#@]/)[0]!
  let dropped = 0
  for (const key of [...entries.keys()]) {
    if (owner(key) !== sessionID) continue
    disposeSession(key)
    dropped += 1
  }
  return dropped
}

/** Drop every realm. Called on a full instance reload: a kernel holds the whole
 * externalized payload, and a reload that leaves realms behind keeps that payload
 * resident with nothing left to address it. */
export function disposeEverything(): number {
  const dropped = entries.size
  for (const key of [...entries.keys()]) disposeSession(key)
  return dropped
}

/** Idle kernels hold up to MEMORY_LIMIT each, so a session that stops using the
 * tool must not keep its realm forever. Swept on acquire and hourly; `unref`
 * keeps the timer from holding the process open. */
function startSweeper(): void {
  sweep ??= setInterval(() => {
    const cutoff = Date.now() - IDLE_DISPOSE_MS
    for (const [sessionID, entry] of entries) if (entry.used < cutoff) disposeSession(sessionID)
  }, 60 * 60 * 1000)
  sweep.unref?.()
}
