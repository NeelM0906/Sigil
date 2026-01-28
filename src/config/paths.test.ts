import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  resolveDefaultConfigCandidates,
  resolveOAuthDir,
  resolveOAuthPath,
  resolveStateDir,
} from "./paths.js";

describe("oauth paths", () => {
  it("prefers SIGIL_OAUTH_DIR over SIGIL_STATE_DIR", () => {
    const env = {
      SIGIL_OAUTH_DIR: "/custom/oauth",
      SIGIL_STATE_DIR: "/custom/state",
    } as NodeJS.ProcessEnv;

    expect(resolveOAuthDir(env, "/custom/state")).toBe(path.resolve("/custom/oauth"));
    expect(resolveOAuthPath(env, "/custom/state")).toBe(
      path.join(path.resolve("/custom/oauth"), "oauth.json"),
    );
  });

  it("derives oauth path from SIGIL_STATE_DIR when unset", () => {
    const env = {
      SIGIL_STATE_DIR: "/custom/state",
    } as NodeJS.ProcessEnv;

    expect(resolveOAuthDir(env, "/custom/state")).toBe(path.join("/custom/state", "credentials"));
    expect(resolveOAuthPath(env, "/custom/state")).toBe(
      path.join("/custom/state", "credentials", "oauth.json"),
    );
  });
});

describe("state + config path candidates", () => {
  it("prefers SIGIL_STATE_DIR over legacy state dir env", () => {
    const env = {
      SIGIL_STATE_DIR: "/new/state",
      CLAWDBOT_STATE_DIR: "/legacy/state",
    } as NodeJS.ProcessEnv;

    expect(resolveStateDir(env, () => "/home/test")).toBe(path.resolve("/new/state"));
  });

  it("orders default config candidates as new then legacy", () => {
    const home = "/home/test";
    const candidates = resolveDefaultConfigCandidates({} as NodeJS.ProcessEnv, () => home);
    expect(candidates[0]).toBe(path.join(home, ".sigil", "sigil.json"));
    expect(candidates[1]).toBe(path.join(home, ".clawdbot", "sigil.json"));
  });
});
