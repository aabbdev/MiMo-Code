import { describe, expect, test } from "bun:test"
import {
  admitsFailure,
  ALL_SIGNALS,
  claimsOf,
  consistentSet,
  electConsistent,
  firstSentence,
  groundingOf,
  normalizeVerdict,
  overlap,
  overlapMatrix,
  previewOf,
  scoreOf,
  select,
  splitVerdict,
  verbalizedConfidence,
  type Candidate,
  type Signals,
} from "../../src/rlm/search"

const candidate = (answer: string, length: number, confidence?: number): Candidate => ({ answer, length, confidence })

describe("claimsOf", () => {
  test("keeps identifiers and drops ordinary words", () => {
    const claims = claimsOf("Conversion/DPUConversion.cpp uses ConvertLinalgToDPUPass and the plan, which is fine")
    expect(claims.has("conversion/dpuconversion.cpp")).toBe(true)
    expect(claims.has("convertlinalgtodpupass")).toBe(true)
    expect(claims.has("plan")).toBe(false) // < 4 chars
    expect(claims.has("which")).toBe(false) // plain word
  })

  test("a namespaced symbol is one claim", () => {
    expect(claimsOf("mlir::PassManager::run").has("mlir::passmanager::run")).toBe(true)
  })

  test("prose with no identifiers yields nothing, which is what makes it incomparable", () => {
    expect(claimsOf("the answer is that it depends on the context").size).toBe(0)
  })
})

describe("overlap", () => {
  test("is the Jaccard ratio, and zero when either side is empty", () => {
    expect(overlap(new Set(["a_b", "c_d"]), new Set(["a_b", "c_d"]))).toBe(1)
    expect(overlap(new Set(["a_b"]), new Set(["c_d"]))).toBe(0)
    expect(overlap(new Set(["a_b", "c_d"]), new Set(["a_b", "e_f"]))).toBeCloseTo(1 / 3, 10)
    expect(overlap(new Set(), new Set(["a_b"]))).toBe(0)
  })
})

describe("consistentSet", () => {
  test("unanimous agreement keeps everyone", () => {
    const claims = ["alpha_beta gamma_delta", "gamma_delta alpha_beta", "alpha_beta gamma_delta"].map(claimsOf)
    expect(consistentSet(claims)).toEqual({ indices: [0, 1, 2], agreed: true })
  })

  test("a majority filters out the dissenter", () => {
    const claims = [claimsOf("alpha_beta gamma_delta"), claimsOf("alpha_beta gamma_delta"), claimsOf("theta_iota kappa_lambda")].map((c) => c)
    const result = consistentSet(claims)
    expect(result.indices).toEqual([0, 1])
    expect(result.agreed).toBe(true)
  })

  test("mutually disjoint answers elect nothing, so nothing is filtered", () => {
    // The bug this guards: each candidate is its own group of one, and taking the
    // largest would silently discard every candidate but the first.
    const claims = [claimsOf("alpha_beta"), claimsOf("gamma_delta"), claimsOf("theta_iota")]
    expect(consistentSet(claims)).toEqual({ indices: [0, 1, 2], agreed: false })
  })

  test("prose with no claims at all is incomparable, not unanimous", () => {
    const claims = [claimsOf("it depends"), claimsOf("hard to say"), claimsOf("cannot tell")]
    expect(consistentSet(claims)).toEqual({ indices: [0, 1, 2], agreed: false })
  })
})

describe("verbalizedConfidence", () => {
  test("aggregates in log space, so a perfectly confident step costs nothing", () => {
    expect(verbalizedConfidence([100, 100])).toBe(0)
    expect(verbalizedConfidence([50])).toBeCloseTo(Math.log(0.5), 12)
  })

  test("ten confident steps beat twenty wavering ones", () => {
    const confident = verbalizedConfidence(Array(10).fill(95))!
    const wavering = verbalizedConfidence(Array(20).fill(60))!
    expect(confident).toBeGreaterThan(wavering)
    expect(confident).toBeLessThanOrEqual(0)
  })

  test("no reports is undefined, which keeps silent distinguishable from confident", () => {
    expect(verbalizedConfidence([])).toBeUndefined()
    expect(verbalizedConfidence([0, -5, Number.NaN, 500])).toBeUndefined()
  })

  test("out-of-range values are dropped rather than poisoning the sum", () => {
    expect(verbalizedConfidence([100, 0, 101])).toBe(0)
  })
})

describe("select", () => {
  test("a trajectory that produced nothing is discarded when another answered", () => {
    const result = select([candidate("", 900, -0.1), candidate("alpha_beta gamma_delta", 500, -0.4)])
    expect(result.chosen).toBe(1)
    expect(result.report[0]!.eligible).toBe(false)
  })

  test("among agreeing candidates the highest VC·Len wins", () => {
    const result = select([
      candidate("alpha_beta gamma_delta", 500, -0.9),
      candidate("alpha_beta gamma_delta", 500, -0.2),
      candidate("alpha_beta gamma_delta", 500, -0.5),
    ])
    expect(result.chosen).toBe(1)
    expect(result.basis).toContain("VC·Len")
  })

  test("a candidate outside the agreeing set cannot win, however it scores", () => {
    const result = select([
      candidate("alpha_beta gamma_delta", 500, -0.9),
      candidate("alpha_beta gamma_delta", 500, -0.8),
      candidate("theta_iota kappa_lambda", 10, -0.01), // short and confident, but alone
    ])
    expect(result.consistent).toEqual([0, 1])
    expect(result.chosen).toBe(1)
  })

  test("a scored candidate beats an unscored one, so silence never wins by default", () => {
    const result = select([candidate("alpha_beta gamma_delta", 50), candidate("alpha_beta gamma_delta", 900, -0.9)])
    expect(result.chosen).toBe(1)
  })

  test("with no confidence anywhere it falls back to length AND says the signal is weak", () => {
    const result = select([candidate("alpha_beta gamma_delta", 900), candidate("alpha_beta gamma_delta", 100)])
    expect(result.chosen).toBe(1)
    // The paper's ablation: length alone does not reliably indicate correctness.
    expect(result.basis).toContain("unreliable")
  })

  test("no answers at all is reported, not guessed", () => {
    const result = select([candidate("", 10), candidate("", 20)])
    expect(result.chosen).toBe(-1)
    expect(result.basis).toContain("no candidate")
  })

  test("the report exposes every candidate's signals for the caller to surface", () => {
    const result = select([candidate("alpha_beta", 100, -0.5), candidate("alpha_beta", 200)])
    expect(result.report).toHaveLength(2)
    expect(result.report[0]).toMatchObject({ index: 0, eligible: true, claims: 1, confidence: -0.5, length: 100 })
    expect(result.report[0]!.score).toBeCloseTo(-50, 10)
    expect(result.report[1]!.score).toBeUndefined()
  })
})

describe("electConsistent", () => {
  const withVerdict = (verdict: string) => ({ answer: "detail", verdict, length: 10 })

  test("verdict comparison ignores case, punctuation and spacing", () => {
    expect(normalizeVerdict("  Yes, the contract HOLDS!  ")).toBe("yes the contract holds")
    expect(normalizeVerdict("Yes — the contract holds")).toBe("yes the contract holds")
  })

  test("exact verdicts elect, which is the paper's out(p) = a and needs no threshold", () => {
    const candidates = [withVerdict("Yes, the contract holds"), withVerdict("yes  the contract holds!"), withVerdict("No")]
    const elected = electConsistent(candidates, candidates.map((c) => claimsOf(c.answer)))
    expect(elected).toEqual({ indices: [0, 1], agreed: true, basis: "verdict" })
  })

  test("all verdicts differing elects nothing rather than picking the first", () => {
    const candidates = [withVerdict("Alpha"), withVerdict("Beta"), withVerdict("Gamma")]
    const elected = electConsistent(candidates, candidates.map((c) => claimsOf(c.answer)))
    expect(elected).toEqual({ indices: [0, 1, 2], agreed: false, basis: "none" })
  })

  test("fewer than two verdicts falls back to claim overlap and says so", () => {
    const candidates = [
      { answer: "alpha_beta gamma_delta", verdict: "Alpha", length: 10 },
      { answer: "alpha_beta gamma_delta", length: 10 },
      { answer: "theta_iota kappa_lambda", length: 10 },
    ]
    const elected = electConsistent(candidates, candidates.map((c) => claimsOf(c.answer)))
    expect(elected.basis).toBe("claims")
    expect(elected.indices).toEqual([0, 1])
  })
})

describe("overlapMatrix and previewOf", () => {
  test("the matrix exposes agreement so `agreed` can be checked, not believed", () => {
    const claims = [claimsOf("alpha_beta gamma_delta"), claimsOf("alpha_beta gamma_delta"), claimsOf("theta_iota")]
    const matrix = overlapMatrix(claims)
    expect(matrix[0]![1]).toBe(1)
    expect(matrix[0]![2]).toBe(0)
    expect(matrix[0]![0]).toBe(1)
    expect(matrix).toHaveLength(3)
    expect(matrix[0]).toHaveLength(3)
  })

  test("a losing candidate stays identifiable after its answer is discarded", () => {
    expect(previewOf({ answer: "long detail", verdict: "Short verdict", length: 1 })).toBe("Short verdict")
    expect(previewOf({ answer: "  a   long \n detail ", length: 1 })).toBe("a long detail")
    expect(previewOf({ answer: "   ", length: 1 })).toBe("(no answer)")
  })
})

describe("content-aware eligibility", () => {
  // The measured failure, as a fixture: the trajectory that gave up is BOTH the
  // most confident and the most concise, so the paper's VC·Len score picks it.
  const quitter = { answer: "I could not produce the requested lines. I only established structural facts.", length: 5057, confidence: -0.11 }
  const worker = { answer: "ConvertLinalgToDPUPass alpha_beta gamma_delta and also a_long_identifier plus another_one", length: 6810, confidence: -0.16 }

  test("the trajectory that gave up does not win, however well it scores", () => {
    expect(scoreOf(quitter)!).toBeGreaterThan(scoreOf(worker)!)
    const result = select([quitter, worker])
    expect(result.chosen).toBe(1)
    expect(result.consistencyBasis).toBe("none")
    expect(result.basis).toContain("most claims")
    expect(result.degraded).toBeUndefined()
  })

  test("with nothing in common, content decides instead of self-report", () => {
    const small = { answer: "alpha_beta gamma_delta", length: 100, confidence: -0.05 }
    const large = { answer: "alpha_beta gamma_delta theta_iota kappa_lambda mu_nu xi_omicron", length: 900, confidence: -0.9 }
    expect(scoreOf(small)!).toBeGreaterThan(scoreOf(large)!)
    // Two claims against six, and the trajectories agree on nothing.
    expect(select([small, large]).chosen).toBe(1)
  })

  test("agreement still wins: the paper's score decides inside a real plurality", () => {
    const agreeing = [
      { answer: "alpha_beta gamma_delta", length: 500, confidence: -0.9 },
      { answer: "alpha_beta gamma_delta", length: 500, confidence: -0.2 },
    ]
    const result = select(agreeing)
    expect(result.consistencyBasis).toBe("claims")
    expect(result.chosen).toBe(1)
    expect(result.basis).toContain("VC·Len")
  })

  test("when nothing carries content the result says so instead of saying ok", () => {
    const result = select([
      { answer: "I could not do it", length: 100, confidence: -0.1 },
      { answer: "Unable to complete the task", length: 200, confidence: -0.2 },
    ])
    expect(result.substantive).toEqual([])
    expect(result.degraded).toContain("no candidate produced substantive content")
  })

  test("a failure admission is detected, but only where one would open with it", () => {
    expect(admitsFailure("I could not produce the requested lines.")).toBe(true)
    expect(admitsFailure("Unable to complete the task")).toBe(true)
    // A good answer may legitimately mention a gap far into its text.
    expect(admitsFailure("Here are the results. " + "x".repeat(700) + " I could not verify one file")).toBe(false)
    expect(admitsFailure("Everything resolved: 41 identifiers verified")).toBe(false)
  })
})

describe("signal ablation (SRLM §3.8)", () => {
  const only = (name: keyof typeof ALL_SIGNALS): Signals => ({
    verdict: name === "verdict",
    claims: name === "claims",
    content: name === "content",
    confidence: name === "confidence",
    length: name === "length",
  })
  // Two trajectories that agree, one confident-and-long, one unsure-and-short.
  const agreeing = [
    { answer: "alpha_beta gamma_delta", length: 500, confidence: -0.9 },
    { answer: "alpha_beta gamma_delta", length: 900, confidence: -0.2 },
  ]

  test("with every signal on, the paper's VC·Len decides", () => {
    const result = select(agreeing, undefined, ALL_SIGNALS)
    expect(result.chosen).toBe(1)
    expect(result.basis).toContain("VC·Len")
  })

  test("length alone picks the shortest, and confidence alone picks the most confident", () => {
    expect(select(agreeing, undefined, only("length")).chosen).toBe(0)
    expect(select(agreeing, undefined, only("confidence")).chosen).toBe(1)
  })

  test("with consensus signals off, nothing is filtered and `basis` says so", () => {
    const result = select(agreeing, undefined, only("confidence"))
    expect(result.consistencyBasis).toBe("none")
    expect(result.consistent).toEqual([0, 1])
  })

  test("with content off, eligibility stops seeing claims — the paper's behaviour, not mine", () => {
    const quitter = { answer: "I could not produce the requested lines.", length: 10, confidence: -0.05 }
    const worker = { answer: "ConvertLinalgToDPUPass alpha_beta gamma_delta a_long_identifier", length: 900, confidence: -0.9 }
    // My content guard discards the quitter; without it the paper's score takes it,
    // which is the measured failure the guard exists for.
    expect(select([quitter, worker], undefined, ALL_SIGNALS).chosen).toBe(1)
    expect(select([quitter, worker], undefined, only("confidence")).chosen).toBe(0)
  })
})

describe("splitVerdict", () => {
  test("a delimited FINAL yields the verdict and the detail apart", () => {
    const split = splitVerdict("The contract holds in ClusterConcurrencyContractValidation ||| Full detail here.")
    expect(split.verdict).toBe("The contract holds in ClusterConcurrencyContractValidation")
    expect(split.answer).toBe("Full detail here.")
  })

  test("no delimiter leaves the whole text as the answer, so the fallback still has something", () => {
    // Measured: DeepSeek-V4.1-Flash ignored the separate-JSON verdict twice. A model
    // that ignores this too must still produce a comparable verdict, not nothing.
    const split = splitVerdict("The contract holds everywhere.")
    expect(split.verdict).toBeUndefined()
    expect(split.answer).toBe("The contract holds everywhere.")
  })

  test("an empty side never destroys the answer", () => {
    expect(splitVerdict(" ||| detail").verdict).toBeUndefined()
    expect(splitVerdict(" ||| detail").answer).toBe("detail")
    expect(splitVerdict("verdict ||| ").answer).toBe("verdict ||| ")
  })

  test("only the first delimiter splits, so a detail may contain one", () => {
    expect(splitVerdict("a ||| b ||| c").answer).toBe("b ||| c")
  })
})

describe("firstSentence and groundingOf", () => {
  test("a first sentence gives plurality something to compare when the model reports no verdict", () => {
    expect(firstSentence("The contract holds in two places. It also covers the RTL.")).toBe("The contract holds in two places.")
    expect(firstSentence("one long clause with no terminator at all")).toBe("one long clause with no terminator at all")
    expect(firstSentence("   ")).toBeUndefined()
    expect(firstSentence("x".repeat(500))?.length).toBe(200)
  })

  test("grounding is the share of claims the payload actually mentions", () => {
    const known = new Set(["alpha_beta", "gamma_delta"])
    expect(groundingOf(new Set(["alpha_beta", "gamma_delta"]), known)).toBe(1)
    expect(groundingOf(new Set(["alpha_beta", "invented_one"]), known)).toBe(0.5)
    expect(groundingOf(new Set(["nowhere_at_all"]), known)).toBe(0)
    // No claims is not a fabrication.
    expect(groundingOf(new Set(), known)).toBe(1)
  })

  test("grounding is REPORTED, never ranked — it cannot tell a read answer from a named one", () => {
    // A name-derived answer cites identifiers that all occur in the payload, which
    // is exactly why grounding was demoted from a ranking signal to a report.
    // The claim forms are lowercased and trailing punctuation is stripped, so the
    // known set has to hold what `claimsOf` actually produces.
    const known = new Set(["commandop", "dpuconversion.cpp"])
    const derived = { answer: "uses CommandOp and Linalg in DPUConversion.cpp", length: 10, confidence: -0.1 }
    const read = { answer: "uses CommandOp and Linalg in DPUConversion.cpp", length: 900, confidence: -0.9 }
    const result = select([derived, read], known)
    expect(result.report[0]!.grounding).toBe(1)
    expect(result.report[1]!.grounding).toBe(1)
    expect(result.degraded).toBeUndefined()
  })

  test("a fabricated answer is flagged, which is the one thing grounding can prove", () => {
    const known = new Set(["real_thing"])
    const fabricated = {
      answer: "alpha_one beta_two gamma_three delta_four epsilon_five zeta_six eta_seven theta_eight iota_nine kappa_ten",
      length: 100,
      confidence: -0.1,
    }
    const result = select([fabricated], known)
    expect(result.report[0]!.grounding).toBe(0)
    expect(result.degraded).toContain("do not occur in the payload")
  })
})

describe("scoreOf", () => {
  test("is VC·Len, and undefined without a confidence", () => {
    expect(scoreOf(candidate("x", 200, -0.25))).toBeCloseTo(-50, 10)
    expect(scoreOf(candidate("x", 200))).toBeUndefined()
  })
})
