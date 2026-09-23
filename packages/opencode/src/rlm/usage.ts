/**
 * What a model call billed for, and what that costs.
 *
 * Its own module because three layers need the same types and the same
 * arithmetic: the sub-call factory prices what it streamed (`host.ts`), the
 * automated loop prices its root turns and its sub-calls separately
 * (`tool/rlm.ts`), and the session kernel prices only its sub-calls
 * (`tool/repl.ts`). Written twice it drifts.
 *
 * The cache split is not a detail: on Together a cached input token costs
 * $0.006/M against $0.30/M uncached, so a cost reported without it is wrong by up
 * to 50×.
 */
export type UsageTally = { input: number; output: number; cacheRead: number }

export type Price = { input: number; output: number; cacheRead: number }

/** Cached tokens bill at the cache rate, the remainder at the input rate. */
export function tallyCost(tally: UsageTally, price: Price): number {
  const billable = Math.max(0, tally.input - tally.cacheRead)
  return (billable * price.input + tally.cacheRead * price.cacheRead + tally.output * price.output) / 1_000_000
}

/** Share of input tokens served from the provider's prefix cache. */
export function hitRate(tally: UsageTally): number {
  return tally.input > 0 ? tally.cacheRead / tally.input : 0
}

export function sumTallies(a: UsageTally, b: UsageTally): UsageTally {
  return { input: a.input + b.input, output: a.output + b.output, cacheRead: a.cacheRead + b.cacheRead }
}

export function zeroTally(): UsageTally {
  return { input: 0, output: 0, cacheRead: 0 }
}

/** The rates to price a tally with, from a resolved model. */
export function priceOf(model: { cost: { input: number; output: number; cache: { read: number } } }): Price {
  return { input: model.cost.input, output: model.cost.output, cacheRead: model.cost.cache.read }
}
