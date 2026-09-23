/**
 * SRLM selection — the self-reflective program search of arXiv 2603.15653.
 *
 * The paper's idea: instead of betting the run on the first context-interaction
 * program the model writes, sample K of them and select by internal uncertainty,
 * with no verifier, no reward model and no labels. Two measured failures in this
 * codebase are exactly what it targets: a first decomposition that answered from
 * filenames instead of reading (run 1), and a trajectory that read everything and
 * then never converged (run 2). Both are *bad candidates* — the paper's framework
 * discards them, where the loop's convergence machinery tried to rescue them.
 *
 * Three signals, each derived from the model's own generation:
 *
 *  - self-consistency: which candidates agree. The paper compares final answers
 *    (`out(p) = a`), which only works on short ones. Our answers run to tens of
 *    thousands of characters of prose, so agreement is measured on the CLAIMS a
 *    candidate makes — the code identifiers it names — rather than on its text.
 *  - verbalized confidence: Σ log(ν/100) over the trajectory's own per-step
 *    `{"confidence": ν}` reports. ≤ 0, closer to zero is more confident.
 *  - reasoning length: tokens spent. Used as the paper defines it.
 *
 * Joint score `VC(p) · Len(p)` maximises toward (confident, concise).
 *
 * One honest limitation, carried in the result rather than hidden: the paper's
 * ablation finds that "high confidence OR short traces alone do not reliably
 * indicate correctness" — only their combination does. So when a model reports no
 * confidence at all, selection falls back to length alone and says so in `basis`
 * rather than presenting a weak signal as a strong one.
 */

/** One trajectory, reduced to what selection needs. */
export type Candidate = {
  /** The final answer, or "" when the trajectory stopped without producing one. */
  answer: string
  /** The one-line verdict the trajectory gave with its final answer.
   *
   * The paper's plurality test is `out(p) = a`, which needs comparable outputs —
   * and a 20 000-character prose answer is not one. Comparing identifier sets
   * with a guessed similarity threshold was my substitute for a mechanism I had
   * prevented from working; a verdict makes the test exact and threshold-free.
   * The detail still rides along in `answer`. */
  verdict?: string
  /** Σ log(ν/100) over the trajectory's own confidence reports; undefined when
   * the model reported none. */
  confidence?: number
  /** Reasoning + output tokens the trajectory spent — the paper's `Len`. */
  length: number
}

export type CandidateReport = {
  index: number
  eligible: boolean
  /** Short identifier of what this candidate claimed, so a loser is still
   * auditable after selection has thrown its answer away. */
  preview: string
  verdict?: string
  claims: number
  /** Share of this candidate's claims that actually appear in the payload.
   *
   * REPORTED, NOT RANKED, and that is a deliberate retreat from an earlier claim
   * of mine. Measured: a name-derived answer's identifiers all occur in the
   * payload — run 1's invented descriptions cited `CommandOp`, `Linalg`,
   * `DPUConversion.cpp`, every one of them present — so grounding scores ~1.0 for
   * an invented answer and for a read one. It detects only outright fabrication,
   * which has not yet occurred here (100 % of cited identifiers existed in every
   * run). What actually discriminated was the claim COUNT, which is what the
   * ranking uses. */
  grounding?: number
  confidence?: number
  length: number
  score?: number
}

export type Selection = {
  /** Index into the candidate array, or -1 when no candidate produced an answer. */
  chosen: number
  /** Candidates that agreed with the plurality — the paper's set S. */
  consistent: number[]
  /** Whether that agreement is a strict majority, as opposed to merely the
   * largest group. Reported because a 2-1 split is not the same evidence as 3-0. */
  agreed: boolean
  /** Which signal formed the consistent set: exact verdicts, claim overlap, or
   * neither (nothing could be filtered). */
  consistencyBasis: "verdict" | "claims" | "none"
  /** Indices carrying a substantive share of the pool's claims. */
  substantive: number[]
  /** Pairwise claim overlap, upper triangle, so `agreed` can be checked rather
   * than believed. */
  overlap: number[][]
  /** How the winner was picked, in the caller's words. */
  basis: string
  /** Set when the outcome must not be presented as a clean answer. */
  degraded?: string
  report: CandidateReport[]
}

/**
 * Which signals selection is allowed to use.
 *
 * The paper's §3.8 ablation reports each signal alone against the combination, and
 * that ablation was NOT reproducible here until these switches existed — the
 * implementation always used everything. `content` is this codebase's own
 * addition (claim count), not one of theirs, and it is the one their framework
 * lacks for long free-text answers.
 */
export type Signals = {
  verdict: boolean
  claims: boolean
  content: boolean
  confidence: boolean
  length: boolean
}

export const ALL_SIGNALS: Signals = { verdict: true, claims: true, content: true, confidence: true, length: true }

/** Separates a verdict from the detail inside `FINAL(...)`. */
export const VERDICT_DELIMITER = "|||"

/**
 * `FINAL(verdict ||| detail)` — the verdict delivered inside the construct the
 * model already produces every time.
 *
 * A separate `{"verdict": …}` line was tried first and abandoned on measurement:
 * twice with DeepSeek-V4.1-Flash, including with the reminder on every step, it
 * produced ZERO verdicts. The asymmetry is the lesson — the model follows
 * "end every reply with `{"confidence": N}`", which is unconditional and repeats
 * each step, but not "give a verdict when you finish", which is conditional and
 * requires it to recognise its own finish line. `FINAL(...)` is unconditional in
 * the same way the confidence line is, so the verdict rides inside it.
 *
 * With no delimiter the whole text is the answer and the caller falls back to the
 * first sentence, so a model that ignores this too still yields something
 * comparable rather than nothing.
 */
export function splitVerdict(text: string): { verdict?: string; answer: string } {
  const at = text.indexOf(VERDICT_DELIMITER)
  if (at === -1) return { answer: text }
  const verdict = text.slice(0, at).trim()
  const detail = text.slice(at + VERDICT_DELIMITER.length).trim()
  return {
    verdict: verdict.length > 0 ? verdict : undefined,
    answer: detail.length > 0 ? detail : text,
  }
}

/** Case, punctuation and spacing are noise when two trajectories mean the same
 * thing by a verdict. */
export function normalizeVerdict(verdict: string): string {
  return verdict
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
}

/**
 * The first sentence of an answer, as a verdict of last resort.
 *
 * The elicited `{"verdict": …}` is better — short, and deliberately self-contained
 * — but it depends on compliance, and a measured run reported ZERO verdicts until
 * a reminder was added. A first sentence is shorter than the whole answer, roughly
 * self-contained, and needs nothing from the model, so exact plurality still has
 * something to compare when the model stays silent.
 */
export function firstSentence(answer: string): string | undefined {
  const flat = answer.trim().replace(/\s+/g, " ")
  if (flat.length === 0) return undefined
  const end = flat.search(/[.!?](\s|$)/)
  const sentence = end === -1 ? flat : flat.slice(0, end + 1)
  return sentence.slice(0, 200)
}

/** Two answers count as "the same answer" at this claim overlap. Below it they
 * are about different things, which for selection is the only question that
 * matters. */
const AGREE_THRESHOLD = 0.6

/** Share of the pool's best claim count at which a candidate counts as carrying
 * content rather than merely text. */
const SUBSTANTIVE_SHARE = 0.2

/** How a trajectory phrases having given up. */
const GIVES_UP = /(could not|couldn't|cannot|can't|unable to|failed to|was not able to|no pude|je n'ai pas pu|impossible de)/i

/**
 * Whether a trajectory reports that it did not do the task.
 *
 * The wording alone is not enough — a good answer can legitimately open with "I
 * could not find X, but …" — so it only counts as having given up when the
 * candidate ALSO holds little of the pool's content. That pairing is what the
 * measured failure looked like: "I could not produce the requested lines",
 * 2 350 characters, 3 claims, chosen over a discarded candidate holding 643.
 */
export function admitsFailure(answer: string): boolean {
  return GIVES_UP.test(answer.slice(0, 600))
}

/**
 * The claims an answer makes: the code identifiers it names, order-insensitive.
 *
 * Comparability is the crux for long answers. Two trajectories that name the same
 * functions and files agree about the payload whatever order or phrasing they
 * used, while two that name disjoint sets do not — and that is exactly what the
 * paper's `out(p) = a` test would capture if the answer were three tokens long.
 * A token is salient when it looks like an identifier rather than an English
 * word: it carries `_`, `::`, an internal capital, or a source-file suffix.
 */
export function claimsOf(answer: string): Set<string> {
  const out = new Set<string>()
  for (const raw of answer.match(/[A-Za-z_][A-Za-z0-9_:.\/-]*/g) ?? []) {
    const token = raw.replace(/[.:/-]+$/, "")
    if (token.length < 4) continue
    const salient =
      token.includes("_") || token.includes("::") || /[a-z][A-Z]/.test(token) || /\.(cpp|h|td|mlir|py)$/.test(token)
    if (salient) out.add(token.toLowerCase())
  }
  return out
}

export function overlap(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0
  let shared = 0
  for (const token of a) if (b.has(token)) shared += 1
  return shared / (a.size + b.size - shared)
}

/**
 * The paper's consistent set S: candidates agreeing with the plurality answer.
 * With no comparable answers at all — prose with no identifiers in it — nothing
 * can be filtered, so every candidate is returned and `agreed` is false. That
 * degeneracy is reported rather than papered over: it means the selection rested
 * entirely on the fine-grained signals.
 */
export function consistentSet(claims: Array<Set<string>>): { indices: number[]; agreed: boolean } {
  const all = claims.map((_, index) => index)
  if (claims.length === 0) return { indices: [], agreed: false }
  if (claims.every((c) => c.size === 0)) return { indices: all, agreed: false }

  // Each candidate's group is everyone it overlaps enough with; the largest group
  // is the plurality. Ties go to the lowest index, so the result is deterministic.
  let best: number[] = []
  for (let i = 0; i < claims.length; i++) {
    const group = all.filter((j) => j === i || overlap(claims[i]!, claims[j]!) >= AGREE_THRESHOLD)
    if (group.length > best.length) best = group
  }
  // A best group of one is not a plurality, it is the absence of one: answers
  // that share nothing cannot elect an answer, so nothing may be filtered out.
  // Without this, mutually disjoint candidates would silently discard all but one.
  if (best.length <= 1) return { indices: all, agreed: false }
  return { indices: best, agreed: best.length * 2 > claims.length }
}

/**
 * Elect the consistent set — the paper's set S.
 *
 * Exact verdicts first, because that IS the paper's test (`out(p) = a`). Only
 * when fewer than two trajectories gave a verdict does this fall back to claim
 * overlap, and the result says which path it took, so a reader knows how much
 * weight `agreed` carries.
 */
export function electConsistent(
  candidates: Candidate[],
  claims: Array<Set<string>>,
  signals: Signals = ALL_SIGNALS,
): { indices: number[]; agreed: boolean; basis: "verdict" | "claims" | "none" } {
  const all = candidates.map((_, index) => index)
  if (candidates.length === 0) return { indices: [], agreed: false, basis: "none" }

  const verdicts = candidates.map((candidate) => (candidate.verdict ? normalizeVerdict(candidate.verdict) : ""))
  const speaking = signals.verdict ? all.filter((index) => verdicts[index]!.length > 0) : []
  if (speaking.length >= 2) {
    let best: number[] = []
    for (const index of speaking) {
      const group = speaking.filter((other) => verdicts[other] === verdicts[index])
      if (group.length > best.length) best = group
    }
    // Every verdict differs: no answer can be elected, so nothing is filtered.
    if (best.length <= 1) return { indices: all, agreed: false, basis: "none" }
    return { indices: best, agreed: best.length * 2 > candidates.length, basis: "verdict" }
  }

  if (!signals.claims) return { indices: all, agreed: false, basis: "none" }
  const byClaims = consistentSet(claims)
  if (byClaims.indices.length === candidates.length && !byClaims.agreed) return { ...byClaims, basis: "none" }
  return { ...byClaims, basis: "claims" }
}

/** Pairwise claim overlap, so `agreed` can be CHECKED rather than believed.
 * Without it, a selection whose losing answers were discarded is an assertion
 * with no evidence behind it. */
export function overlapMatrix(claims: Array<Set<string>>): number[][] {
  return claims.map((row) => claims.map((other) => Number(overlap(row, other).toFixed(2))))
}

/** Share of a candidate's claims that appear in the payload's own identifiers.
 * An answer that names things the payload never mentions is fabricating, whatever
 * it says about them. */
export function groundingOf(claims: Set<string>, known: Set<string>): number {
  if (claims.size === 0) return 1
  let present = 0
  for (const claim of claims) if (known.has(claim)) present += 1
  return present / claims.size
}

/** `s(p) = VC(p) · Len(p)`. Both factors are ≤ 0 and > 0 respectively, so the
 * score is ≤ 0 and the maximum is the value closest to zero. */
export function scoreOf(candidate: Candidate): number | undefined {
  if (candidate.confidence === undefined) return undefined
  return candidate.confidence * candidate.length
}

/** A compact label for a candidate, so a loser stays identifiable after its
 * answer has been discarded. */
export function previewOf(candidate: Candidate): string {
  if (candidate.verdict) return candidate.verdict.slice(0, 100)
  const answer = candidate.answer.trim().replace(/\s+/g, " ")
  return answer.length > 0 ? answer.slice(0, 80) : "(no answer)"
}

/**
 * Pick a winner. Trajectories that produced nothing are ineligible whenever any
 * candidate did produce an answer — discarding them is the point of the search.
 * Among the eligible and consistent, a scored candidate beats an unscored one and
 * the highest score wins; with no scores anywhere, the shortest wins and `basis`
 * records that this is the weak signal the paper warns about.
 */
export function select(candidates: Candidate[], known?: Set<string>, signals: Signals = ALL_SIGNALS): Selection {
  const all = candidates.map((_, index) => index)
  const answers = candidates.map((candidate) => candidate.answer.trim())
  const anyAnswer = answers.some((answer) => answer.length > 0)
  // With no answer anywhere there is nothing to select: every candidate stays
  // ineligible so the result says so, instead of electing the least-bad failure.
  const eligible = candidates.map((_, index) => anyAnswer && answers[index]!.length > 0)
  const claims = answers.map((answer) => claimsOf(answer))
  const { indices, agreed, basis: consistencyBasis } = electConsistent(candidates, claims, signals)
  const hasPlurality = consistencyBasis !== "none"

  const maxClaims = claims.reduce((max, set) => Math.max(max, set.size), 0)
  // CONTENT, not self-report. A candidate carries content when it holds a
  // meaningful share of the pool's claims; with no claims anywhere — prose answers
  // — this list is empty and filters nothing, so the heuristic costs nothing where
  // it does not apply.
  const substantive = signals.content
    ? all.filter((index) => maxClaims > 0 && claims[index]!.size >= Math.max(1, maxClaims * SUBSTANTIVE_SHARE))
    : []

  const report: CandidateReport[] = candidates.map((candidate, index) => ({
    index,
    eligible: eligible[index]!,
    preview: previewOf(candidate),
    verdict: candidate.verdict,
    claims: claims[index]!.size,
    grounding: known ? groundingOf(claims[index]!, known) : undefined,
    confidence: candidate.confidence,
    length: candidate.length,
    score: scoreOf(candidate),
  }))
  const overlap = overlapMatrix(claims)

  const consistent = indices.filter((index) => eligible[index])
  const eligiblePool = consistent.length > 0 ? consistent : all.filter((index) => eligible[index])
  if (eligiblePool.length === 0) {
    return {
      chosen: -1,
      consistent: indices,
      agreed,
      consistencyBasis,
      substantive,
      overlap,
      basis: "no candidate produced an answer",
      report,
    }
  }
  // When any candidate carries content, the choice is made among those. This is
  // the correction for the measured failure: the selection returned "I could not
  // produce the requested lines" while a discarded candidate held 643 claims.
  const withContent = eligiblePool.filter((index) => substantive.includes(index))
  const shortlist = withContent.length > 0 ? withContent : eligiblePool

  /**
   * The paper's `s = VC·Len` rewards the confident AND the concise, and a
   * trajectory that gave up early is both at once — measured, the quitter won on
   * both factors because its trace was the shortest and its self-report the
   * highest. The score is therefore trusted only when the trajectories actually
   * agreed on something; with nothing in common, self-reports are uninformative
   * and content decides.
   */
  const order = (a: number, b: number): number => {
    if (signals.content && !hasPlurality) {
      const byContent = report[b]!.claims - report[a]!.claims
      if (byContent !== 0) return byContent
    }
    // `s = VC·Len` needs BOTH factors; with only one of them enabled the ordering
    // falls back to that one alone, which is what an ablation of the pair means.
    if (signals.confidence && signals.length) {
      const scoreA = report[a]!.score
      const scoreB = report[b]!.score
      if (scoreA !== undefined && scoreB !== undefined && scoreA !== scoreB) return scoreB - scoreA
      if (scoreA !== undefined && scoreB === undefined) return -1
      if (scoreA === undefined && scoreB !== undefined) return 1
    }
    if (signals.confidence && !signals.length) {
      const confidenceA = report[a]!.confidence
      const confidenceB = report[b]!.confidence
      if (confidenceA !== undefined && confidenceB !== undefined && confidenceA !== confidenceB) return confidenceB - confidenceA
    }
    if (signals.length) return report[a]!.length - report[b]!.length
    return 0
  }
  const chosen = [...shortlist].sort(order)[0]!

  const basis = !hasPlurality
    ? "most claims — no trajectory agreed with another, so the self-reported signals carry no information"
    : report[chosen]!.score !== undefined
      ? "highest VC·Len among agreeing candidates"
      : "shortest reasoning, no confidence reported — the paper's ablation finds length alone unreliable, treat this choice with suspicion"

  // Nothing substantive anywhere means no candidate actually did the work, and
  // saying "ok" over the least-bad of them would be the defect this whole
  // accounting exists to remove. The same applies to a chosen answer whose
  // identifiers the payload never mentions: that is fabrication, and it is the one
  // thing grounding can prove.
  const chosenRow = report[chosen]!
  const fabricated = chosenRow.claims >= 10 && chosenRow.grounding !== undefined && chosenRow.grounding < 0.5
  const degraded =
    signals.content && substantive.length === 0
      ? `no candidate produced substantive content (best claim count ${maxClaims}); the returned text is the least-bad of ${candidates.length}`
      : fabricated
        ? `the selected answer cites identifiers that do not occur in the payload (${Math.round((chosenRow.grounding ?? 0) * 100)}% of ${chosenRow.claims} appear)`
        : undefined

  return { chosen, consistent: indices, agreed, consistencyBasis, substantive, overlap, basis, degraded, report }
}

/**
 * Aggregate a trajectory's own `{"confidence": ν}` reports into `VC(p)`.
 *
 * Log-space, as the paper defines it: `VC = Σ log(ν/100) ≤ 0`, so a trajectory
 * that stays confident across many steps is penalised less than one that
 * wavers — ten steps at 95 % score better than twenty at 60 %. Returns undefined
 * for no reports, which keeps "silent" distinguishable from "confident".
 */
export function verbalizedConfidence(reports: number[]): number | undefined {
  const usable = reports.filter((value) => Number.isFinite(value) && value > 0 && value <= 100)
  if (usable.length === 0) return undefined
  return usable.reduce((total, value) => total + Math.log(value / 100), 0)
}
