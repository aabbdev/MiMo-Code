import { describe, expect, test } from "bun:test"
import { inflateSync } from "node:zlib"
import { estimateSnap, renderTextImage, shouldSnap, snapDataUrl } from "../../src/session/snapcompact"

/**
 * Minimal PNG decoder for the test only: walks the chunks and inflates the
 * IDAT. The renderer writes filter 0 on every scanline, so unfiltering is a
 * straight copy. Keeping this in-tree avoids leaning on a (transitive) PNG
 * dependency just to read back our own output.
 */
function decode(png: Buffer) {
  let pos = 8
  let width = 0
  let height = 0
  const idat: Buffer[] = []
  while (pos < png.length) {
    const length = png.readUInt32BE(pos)
    const type = png.subarray(pos + 4, pos + 8).toString("ascii")
    const data = png.subarray(pos + 8, pos + 8 + length)
    if (type === "IHDR") {
      width = data.readUInt32BE(0)
      height = data.readUInt32BE(4)
    } else if (type === "IDAT") {
      idat.push(Buffer.from(data))
    } else if (type === "IEND") break
    pos += 12 + length
  }
  const raw = inflateSync(Buffer.concat(idat))
  const stride = width * 4
  const pixels = Buffer.alloc(stride * height)
  for (let y = 0; y < height; y++) {
    expect(raw[y * (stride + 1)]).toBe(0)
    raw.copy(pixels, y * stride, y * (stride + 1) + 1, y * (stride + 1) + 1 + stride)
  }
  return { width, height, data: pixels }
}

describe("snapcompact renderer", () => {
  test("emits a structurally valid PNG with the reported geometry", () => {
    const image = renderTextImage("HELLO\nWORLD", { maxEdge: 160 })
    expect(image.png.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a")
    const decoded = decode(image.png)
    expect(decoded.width).toBe(image.width)
    expect(decoded.height).toBe(image.height)
  })

  test("draws black ink on a white ground", () => {
    const decoded = decode(renderTextImage("H", { maxEdge: 160 }).png)
    let black = 0
    let white = 0
    for (let i = 0; i < decoded.data.length; i += 4) {
      if (decoded.data[i] === 0) black += 1
      else if (decoded.data[i] === 255) white += 1
    }
    // A blank render has zero ink; a fully inked one has no ground.
    expect(black).toBeGreaterThan(0)
    expect(white).toBeGreaterThan(black)
  })

  test("is byte-deterministic for the same input", () => {
    const a = renderTextImage("deterministic block", { maxEdge: 160 })
    const b = renderTextImage("deterministic block", { maxEdge: 160 })
    expect(a.png.equals(b.png)).toBe(true)
  })

  test("scales the cell without changing the geometry contract", () => {
    const tall = renderTextImage("A", { maxEdge: 160, scaleX: 1, scaleY: 2 })
    const flat = renderTextImage("A", { maxEdge: 160, scaleX: 1, scaleY: 1 })
    expect(tall.height).toBeGreaterThan(flat.height)
    expect(decode(tall.png).width).toBe(tall.width)
  })

  test("reports carrying capacity and truncation honestly", () => {
    const overflow = renderTextImage("x".repeat(20_000), { maxEdge: 160 })
    expect(overflow.chars).toBeLessThanOrEqual(overflow.columns * overflow.rows)
    expect(overflow.truncated).toBe(true)

    const fits = renderTextImage("small", { maxEdge: 160 })
    expect(fits.truncated).toBe(false)
    expect(fits.chars).toBe("small".length)
  })

  test("data url wraps the same bytes as a base64 PNG", () => {
    const image = renderTextImage("ok", { maxEdge: 160 })
    const url = snapDataUrl(image)
    expect(url.startsWith("data:image/png;base64,")).toBe(true)
    expect(Buffer.from(url.slice("data:image/png;base64,".length), "base64").equals(image.png)).toBe(true)
  })
})

describe("snapcompact economics", () => {
  test("a short block is a LOSS (an image is billed by area, not by content)", () => {
    // Measured on Together/DeepSeek: a two-line render cost ~239 prompt tokens
    // against ~28 for the same text.
    const economics = estimateSnap("two short lines\nof text", { maxEdge: 1568 })
    expect(economics.imageTokens).toBeGreaterThan(economics.textTokens)
    expect(shouldSnap("two short lines\nof text", { maxEdge: 1568 })).toBe(false)
  })

  test("a filled canvas is a WIN, but the 8x16 cell only clears ~1.5x", () => {
    // Measured shape: a full 1568x1552 canvas carries 19,000 chars (~4,750 text
    // tokens) for ~3,245 image tokens at 750 px/token — a 1.46x win, which the
    // default safety margin (1.5) deliberately refuses.
    const dense = "x".repeat(19_000)
    const tall = estimateSnap(dense, { maxEdge: 1568, scaleY: 2 })
    expect(tall.ratio).toBeGreaterThan(1)
    expect(tall.ratio).toBeLessThan(1.5)
    expect(shouldSnap(dense, { maxEdge: 1568, scaleY: 2 })).toBe(false)

    // A denser cell (8x8) roughly doubles the carrying capacity per image token,
    // which is what clears the margin.
    const flat = estimateSnap(dense, { maxEdge: 1568, scaleY: 1 })
    expect(flat.ratio).toBeGreaterThan(2)
    expect(shouldSnap(dense, { maxEdge: 1568, scaleY: 1 })).toBe(true)
  })
})
