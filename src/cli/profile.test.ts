import path from "node:path";
import { describe, expect, it } from "vitest";
import { formatCliCommand } from "./command-format.js";
import { applyCliProfileEnv, parseCliProfileArgs } from "./profile.js";

describe("parseCliProfileArgs", () => {
  it("leaves gateway --dev for subcommands", () => {
    const res = parseCliProfileArgs(["node", "sigil", "gateway", "--dev", "--allow-unconfigured"]);
    if (!res.ok) throw new Error(res.error);
    expect(res.profile).toBeNull();
    expect(res.argv).toEqual(["node", "sigil", "gateway", "--dev", "--allow-unconfigured"]);
  });

  it("still accepts global --dev before subcommand", () => {
    const res = parseCliProfileArgs(["node", "sigil", "--dev", "gateway"]);
    if (!res.ok) throw new Error(res.error);
    expect(res.profile).toBe("dev");
    expect(res.argv).toEqual(["node", "sigil", "gateway"]);
  });

  it("parses --profile value and strips it", () => {
    const res = parseCliProfileArgs(["node", "sigil", "--profile", "work", "status"]);
    if (!res.ok) throw new Error(res.error);
    expect(res.profile).toBe("work");
    expect(res.argv).toEqual(["node", "sigil", "status"]);
  });

  it("rejects missing profile value", () => {
    const res = parseCliProfileArgs(["node", "sigil", "--profile"]);
    expect(res.ok).toBe(false);
  });

  it("rejects combining --dev with --profile (dev first)", () => {
    const res = parseCliProfileArgs(["node", "sigil", "--dev", "--profile", "work", "status"]);
    expect(res.ok).toBe(false);
  });

  it("rejects combining --dev with --profile (profile first)", () => {
    const res = parseCliProfileArgs(["node", "sigil", "--profile", "work", "--dev", "status"]);
    expect(res.ok).toBe(false);
  });
});

describe("applyCliProfileEnv", () => {
  it("fills env defaults for dev profile", () => {
    const env: Record<string, string | undefined> = {};
    applyCliProfileEnv({
      profile: "dev",
      env,
      homedir: () => "/home/peter",
    });
    const expectedStateDir = path.join("/home/peter", ".sigil-dev");
    expect(env.SIGIL_PROFILE).toBe("dev");
    expect(env.SIGIL_STATE_DIR).toBe(expectedStateDir);
    expect(env.SIGIL_CONFIG_PATH).toBe(path.join(expectedStateDir, "sigil.json"));
    expect(env.SIGIL_GATEWAY_PORT).toBe("19001");
  });

  it("does not override explicit env values", () => {
    const env: Record<string, string | undefined> = {
      SIGIL_STATE_DIR: "/custom",
      SIGIL_GATEWAY_PORT: "19099",
    };
    applyCliProfileEnv({
      profile: "dev",
      env,
      homedir: () => "/home/peter",
    });
    expect(env.SIGIL_STATE_DIR).toBe("/custom");
    expect(env.SIGIL_GATEWAY_PORT).toBe("19099");
    expect(env.SIGIL_CONFIG_PATH).toBe(path.join("/custom", "sigil.json"));
  });
});

describe("formatCliCommand", () => {
  it("returns command unchanged when no profile is set", () => {
    expect(formatCliCommand("sigil doctor --fix", {})).toBe("sigil doctor --fix");
  });

  it("returns command unchanged when profile is default", () => {
    expect(formatCliCommand("sigil doctor --fix", { SIGIL_PROFILE: "default" })).toBe(
      "sigil doctor --fix",
    );
  });

  it("returns command unchanged when profile is Default (case-insensitive)", () => {
    expect(formatCliCommand("sigil doctor --fix", { SIGIL_PROFILE: "Default" })).toBe(
      "sigil doctor --fix",
    );
  });

  it("returns command unchanged when profile is invalid", () => {
    expect(formatCliCommand("sigil doctor --fix", { SIGIL_PROFILE: "bad profile" })).toBe(
      "sigil doctor --fix",
    );
  });

  it("returns command unchanged when --profile is already present", () => {
    expect(formatCliCommand("sigil --profile work doctor --fix", { SIGIL_PROFILE: "work" })).toBe(
      "sigil --profile work doctor --fix",
    );
  });

  it("returns command unchanged when --dev is already present", () => {
    expect(formatCliCommand("sigil --dev doctor", { SIGIL_PROFILE: "dev" })).toBe(
      "sigil --dev doctor",
    );
  });

  it("inserts --profile flag when profile is set", () => {
    expect(formatCliCommand("sigil doctor --fix", { SIGIL_PROFILE: "work" })).toBe(
      "sigil --profile work doctor --fix",
    );
  });

  it("trims whitespace from profile", () => {
    expect(formatCliCommand("sigil doctor --fix", { SIGIL_PROFILE: "  jbclawd  " })).toBe(
      "sigil --profile jbclawd doctor --fix",
    );
  });

  it("handles command with no args after sigil", () => {
    expect(formatCliCommand("sigil", { SIGIL_PROFILE: "test" })).toBe("sigil --profile test");
  });

  it("handles pnpm wrapper", () => {
    expect(formatCliCommand("pnpm sigil doctor", { SIGIL_PROFILE: "work" })).toBe(
      "pnpm sigil --profile work doctor",
    );
  });
});
