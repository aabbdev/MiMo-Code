/**
 * Snap compaction — carry a large text context as a dense pixel-font PNG.
 *
 * The provider bills an image by its pixel area, not by the characters it
 * renders inside: a 1568² PNG holds tens of thousands of characters of 8×8
 * bitmap text while billing a fraction of the equivalent text tokens, and
 * vision models read bitmap text back near-verbatim. So at a moment when the
 * prompt cache is ALREADY being invalidated (an epoch boundary), a big block can
 * be handed over as an image instead of as text.
 *
 * Scope of THIS module: the renderer only. It is deliberately pure and
 * dependency-free (node:zlib for the PNG deflate, nothing else) so it can be
 * unit-tested by decoding its own output, and so wiring it into a request is a
 * separate, reviewable decision.
 *
 * Two hazards the caller must respect, both measured against this repo:
 *   1. VISION. Only send the image to a model whose `capabilities.input.image`
 *      is true. `ProviderTransform.unsupportedParts` otherwise rewrites the part
 *      into an "ERROR: Cannot read image" text block, so a text-only model gets
 *      a poisoned prompt instead of the content.
 *   2. THE IMAGE CAP. `ProviderTransform.limitImages` downscales images above
 *      `DEFAULT_MAX_IMAGE_DIMENSION` for Anthropic/Bedrock routes. A downscale
 *      destroys the cell alignment this renderer depends on, so either stay
 *      under the cap or skip snap for those providers.
 */
import { deflateSync } from "node:zlib"
import { FONT8X8_BASIC } from "./snapcompact-font"

export type SnapImage = {
  /** 8-bit RGBA PNG, ready to be attached as an image part. */
  png: Buffer
  width: number
  height: number
  /** Characters per line, and lines per image — the carrying capacity. */
  columns: number
  rows: number
  /** Characters actually rendered (the tail past capacity is dropped). */
  chars: number
  /** True when `text` did not fit and was truncated to `chars`. */
  truncated: boolean
}

export type SnapOptions = {
  /** Glyph width multiplier: 1 renders 8px-wide cells. */
  scaleX?: number
  /** Glyph height multiplier: 2 renders 16px-tall rows (the "8 on 16" cell). */
  scaleY?: number
  /** Maximum image edge, in px. Defaults to 1568 (a common vision cap). */
  maxEdge?: number
}

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

const CRC_TABLE = (() => {
  const table = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c >>> 0
  }
  return table
})()

function crc32(input: Buffer): number {
  let c = 0xffffffff
  for (const byte of input) c = (CRC_TABLE[(c ^ byte) & 0xff] ?? 0) ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function chunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4)
  length.writeUInt32BE(data.length, 0)
  const body = Buffer.concat([Buffer.from(type, "ascii"), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body), 0)
  return Buffer.concat([length, body, crc])
}

function encodePng(width: number, height: number, rgba: Buffer): Buffer {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 6 // colour type: RGBA
  const stride = width * 4
  const raw = Buffer.alloc((stride + 1) * height)
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0 // per-scanline filter: none
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride)
  }
  return Buffer.concat([
    PNG_SIGNATURE,
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ])
}

/** Hard-wrap into `columns`-wide lines, capped at `rows` lines. */
function wrap(text: string, columns: number, rows: number): { lines: string[]; truncated: boolean } {
  const lines: string[] = []
  let truncated = false
  for (const raw of text.replace(/\r\n?/g, "\n").split("\n")) {
    if (lines.length >= rows) {
      truncated = true
      break
    }
    if (raw.length === 0) {
      lines.push("")
      continue
    }
    for (let i = 0; i < raw.length; i += columns) {
      if (lines.length >= rows) {
        truncated = true
        break
      }
      lines.push(raw.slice(i, i + columns))
    }
    if (truncated) break
  }
  return { lines, truncated }
}

/**
 * Render `text` into a black-on-white pixel-font PNG.
 *
 * Deterministic: the same input and options always produce byte-identical PNG
 * bytes, so a rendered block can be cached against its source hash.
 */
export function renderTextImage(text: string, options: SnapOptions = {}): SnapImage {
  const scaleX = Math.max(1, Math.floor(options.scaleX ?? 1))
  const scaleY = Math.max(1, Math.floor(options.scaleY ?? 2))
  const maxEdge = Math.max(16, Math.floor(options.maxEdge ?? 1568))
  const cellWidth = 8 * scaleX
  const cellHeight = 8 * scaleY
  const columns = Math.max(1, Math.floor(maxEdge / cellWidth))
  const rows = Math.max(1, Math.floor(maxEdge / cellHeight))

  const { lines, truncated } = wrap(text, columns, rows)
  // Height fits the CONTENT, not the row capacity: blank rows below the text
  // bill as image tokens and carry nothing. Width stays at the full column count
  // so every line keeps the same cell alignment.
  const width = columns * cellWidth
  const height = Math.max(1, lines.length) * cellHeight
  const pixels = Buffer.alloc(width * height * 4, 0xff) // white

  let chars = 0
  for (let row = 0; row < lines.length; row++) {
    const line = lines[row] ?? ""
    for (let col = 0; col < line.length; col++) {
      const glyph = FONT8X8_BASIC[line.charCodeAt(col)]
      if (!glyph) continue
      chars += 1
      for (let gy = 0; gy < 8; gy++) {
        const bits = glyph[gy] ?? 0
        if (bits === 0) continue
        for (let gx = 0; gx < 8; gx++) {
          if ((bits & (1 << gx)) === 0) continue
          for (let sy = 0; sy < scaleY; sy++) {
            for (let sx = 0; sx < scaleX; sx++) {
              const x = col * cellWidth + gx * scaleX + sx
              const y = row * cellHeight + gy * scaleY + sy
              const offset = (y * width + x) * 4
              pixels[offset] = 0
              pixels[offset + 1] = 0
              pixels[offset + 2] = 0
            }
          }
        }
      }
    }
  }

  return { png: encodePng(width, height, pixels), width, height, columns, rows, chars, truncated }
}

/** `data:` URL for the rendered image, for providers that take inline images. */
export function snapDataUrl(image: SnapImage): string {
  return `data:image/png;base64,${image.png.toString("base64")}`
}

export type SnapEconomics = {
  /** Rough token cost of the same block as plain text (~4 chars/token). */
  textTokens: number
  /** Token cost of the rendered image, from its pixel AREA. */
  imageTokens: number
  /** textTokens / imageTokens. > 1 means snapping is cheaper. */
  ratio: number
}

/**
 * Economics of snapping `text`. An image is billed by pixel AREA, so snapping a
 * SHORT block is a LOSS — measured on Together/DeepSeek, a two-line render
 * (1568×32) cost ~239 prompt tokens against ~28 for the same text. Snap only
 * pays when the canvas is filled.
 *
 * `pixelsPerToken` is provider-specific and must be calibrated: Anthropic's
 * formula is ~750 px/token, while GPT-5.5 charges roughly a flat ~2.9k tokens
 * per image regardless of area. 750 is the sane default for a first estimate.
 * `shouldSnap` therefore returns false unless the caller has confirmed the
 * provider's real ratio.
 */
export function estimateSnap(text: string, options: SnapOptions & { pixelsPerToken?: number } = {}): SnapEconomics {
  const image = renderTextImage(text, options)
  const pixelsPerToken = Math.max(1, options.pixelsPerToken ?? 750)
  const textTokens = Math.ceil(text.length / 4)
  const imageTokens = Math.ceil((image.width * image.height) / pixelsPerToken)
  return { textTokens, imageTokens, ratio: imageTokens > 0 ? textTokens / imageTokens : 0 }
}

/**
 * True when snapping is expected to be cheaper than sending the text — with a
 * safety margin, because the decode tax (vision models spend extra output
 * tokens reading the image) is not counted here.
 */
export function shouldSnap(text: string, options: SnapOptions & { pixelsPerToken?: number; margin?: number } = {}) {
  const economics = estimateSnap(text, options)
  return economics.ratio >= (options.margin ?? 1.5)
}
