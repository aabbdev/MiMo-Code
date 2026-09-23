import { describe, expect, test } from "bun:test"
import { forkArchive, upgradeRepo } from "../../src/installation"

describe("MIMOCODE_UPGRADE_REPO", () => {
  // The defect this exists for, measured twice: this binary is a build of a FORK,
  // and the upstream release sources cannot serve it — an upstream PATCH installs
  // itself on any invocation (`cli/upgrade.ts`), the fork-only config keys
  // (`session`, `checkpoint`) then read as "Unrecognized key", and the install
  // stops starting.

  test("unset means every source stays exactly as it was", () => {
    expect(upgradeRepo({})).toBeUndefined()
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "" })).toBeUndefined()
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "   " })).toBeUndefined()
  })

  test("takes a clone URL as readily as owner/name, because both are what people paste", () => {
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "aabbdev/MiMo-Code" })).toBe("aabbdev/MiMo-Code")
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "https://github.com/aabbdev/MiMo-Code" })).toBe("aabbdev/MiMo-Code")
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "git@github.com:aabbdev/MiMo-Code.git" })).toBeUndefined()
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "https://github.com/aabbdev/MiMo-Code.git" })).toBe("aabbdev/MiMo-Code")
  })

  test("anything that is not owner/name is refused rather than turned into a URL", () => {
    // A half-parsed repo would fetch from github.com/<garbage> and the failure would
    // surface as an unexplained 404 at upgrade time.
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "MiMo-Code" })).toBeUndefined()
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "a/b/c" })).toBeUndefined()
    expect(upgradeRepo({ MIMOCODE_UPGRADE_REPO: "https://gitlab.com/a/b" })).toBeUndefined()
  })
})

describe("forkArchive", () => {
  test("names the archive the build script publishes, per platform", () => {
    expect(forkArchive("linux", "x64")).toBe("mimocode-linux-x64.tar.gz")
    expect(forkArchive("linux", "arm64")).toBe("mimocode-linux-arm64.tar.gz")
    expect(forkArchive("darwin", "arm64")).toBe("mimocode-darwin-arm64.zip")
    expect(forkArchive("win32", "x64")).toBe("mimocode-windows-x64.zip")
  })

  test("an unknown arch falls back to x64 rather than inventing a name", () => {
    expect(forkArchive("linux", "ia32")).toBe("mimocode-linux-x64.tar.gz")
  })
})
