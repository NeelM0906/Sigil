import path from "node:path";

import { describe, expect, it } from "vitest";

import { resolveGatewayStateDir } from "./paths.js";

describe("resolveGatewayStateDir", () => {
  it("uses the default state dir when no overrides are set", () => {
    const env = { HOME: "/Users/test" };
    expect(resolveGatewayStateDir(env)).toBe(path.join("/Users/test", ".sigil"));
  });

  it("appends the profile suffix when set", () => {
    const env = { HOME: "/Users/test", SIGIL_PROFILE: "rescue" };
    expect(resolveGatewayStateDir(env)).toBe(path.join("/Users/test", ".sigil-rescue"));
  });

  it("treats default profiles as the base state dir", () => {
    const env = { HOME: "/Users/test", SIGIL_PROFILE: "Default" };
    expect(resolveGatewayStateDir(env)).toBe(path.join("/Users/test", ".sigil"));
  });

  it("uses SIGIL_STATE_DIR when provided", () => {
    const env = { HOME: "/Users/test", SIGIL_STATE_DIR: "/var/lib/sigil" };
    expect(resolveGatewayStateDir(env)).toBe(path.resolve("/var/lib/sigil"));
  });

  it("expands ~ in SIGIL_STATE_DIR", () => {
    const env = { HOME: "/Users/test", SIGIL_STATE_DIR: "~/sigil-state" };
    expect(resolveGatewayStateDir(env)).toBe(path.resolve("/Users/test/sigil-state"));
  });

  it("preserves Windows absolute paths without HOME", () => {
    const env = { SIGIL_STATE_DIR: "C:\\State\\sigil" };
    expect(resolveGatewayStateDir(env)).toBe("C:\\State\\sigil");
  });
});
