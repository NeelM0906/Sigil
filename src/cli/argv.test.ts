import { describe, expect, it } from "vitest";

import {
  buildParseArgv,
  getFlagValue,
  getCommandPath,
  getPrimaryCommand,
  getPositiveIntFlagValue,
  getVerboseFlag,
  hasHelpOrVersion,
  hasFlag,
  shouldMigrateState,
  shouldMigrateStateFromPath,
} from "./argv.js";

describe("argv helpers", () => {
  it("detects help/version flags", () => {
    expect(hasHelpOrVersion(["node", "sigil", "--help"])).toBe(true);
    expect(hasHelpOrVersion(["node", "sigil", "-V"])).toBe(true);
    expect(hasHelpOrVersion(["node", "sigil", "status"])).toBe(false);
  });

  it("extracts command path ignoring flags and terminator", () => {
    expect(getCommandPath(["node", "sigil", "status", "--json"], 2)).toEqual(["status"]);
    expect(getCommandPath(["node", "sigil", "agents", "list"], 2)).toEqual(["agents", "list"]);
    expect(getCommandPath(["node", "sigil", "status", "--", "ignored"], 2)).toEqual(["status"]);
  });

  it("returns primary command", () => {
    expect(getPrimaryCommand(["node", "sigil", "agents", "list"])).toBe("agents");
    expect(getPrimaryCommand(["node", "sigil"])).toBeNull();
  });

  it("parses boolean flags and ignores terminator", () => {
    expect(hasFlag(["node", "sigil", "status", "--json"], "--json")).toBe(true);
    expect(hasFlag(["node", "sigil", "--", "--json"], "--json")).toBe(false);
  });

  it("extracts flag values with equals and missing values", () => {
    expect(getFlagValue(["node", "sigil", "status", "--timeout", "5000"], "--timeout")).toBe(
      "5000",
    );
    expect(getFlagValue(["node", "sigil", "status", "--timeout=2500"], "--timeout")).toBe("2500");
    expect(getFlagValue(["node", "sigil", "status", "--timeout"], "--timeout")).toBeNull();
    expect(getFlagValue(["node", "sigil", "status", "--timeout", "--json"], "--timeout")).toBe(
      null,
    );
    expect(getFlagValue(["node", "sigil", "--", "--timeout=99"], "--timeout")).toBeUndefined();
  });

  it("parses verbose flags", () => {
    expect(getVerboseFlag(["node", "sigil", "status", "--verbose"])).toBe(true);
    expect(getVerboseFlag(["node", "sigil", "status", "--debug"])).toBe(false);
    expect(getVerboseFlag(["node", "sigil", "status", "--debug"], { includeDebug: true })).toBe(
      true,
    );
  });

  it("parses positive integer flag values", () => {
    expect(getPositiveIntFlagValue(["node", "sigil", "status"], "--timeout")).toBeUndefined();
    expect(
      getPositiveIntFlagValue(["node", "sigil", "status", "--timeout"], "--timeout"),
    ).toBeNull();
    expect(
      getPositiveIntFlagValue(["node", "sigil", "status", "--timeout", "5000"], "--timeout"),
    ).toBe(5000);
    expect(
      getPositiveIntFlagValue(["node", "sigil", "status", "--timeout", "nope"], "--timeout"),
    ).toBeUndefined();
  });

  it("builds parse argv from raw args", () => {
    const nodeArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["node", "sigil", "status"],
    });
    expect(nodeArgv).toEqual(["node", "sigil", "status"]);

    const versionedNodeArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["node-22", "sigil", "status"],
    });
    expect(versionedNodeArgv).toEqual(["node-22", "sigil", "status"]);

    const versionedNodeWindowsArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["node-22.2.0.exe", "sigil", "status"],
    });
    expect(versionedNodeWindowsArgv).toEqual(["node-22.2.0.exe", "sigil", "status"]);

    const versionedNodePatchlessArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["node-22.2", "sigil", "status"],
    });
    expect(versionedNodePatchlessArgv).toEqual(["node-22.2", "sigil", "status"]);

    const versionedNodeWindowsPatchlessArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["node-22.2.exe", "sigil", "status"],
    });
    expect(versionedNodeWindowsPatchlessArgv).toEqual(["node-22.2.exe", "sigil", "status"]);

    const versionedNodeWithPathArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["/usr/bin/node-22.2.0", "sigil", "status"],
    });
    expect(versionedNodeWithPathArgv).toEqual(["/usr/bin/node-22.2.0", "sigil", "status"]);

    const nodejsArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["nodejs", "sigil", "status"],
    });
    expect(nodejsArgv).toEqual(["nodejs", "sigil", "status"]);

    const nonVersionedNodeArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["node-dev", "sigil", "status"],
    });
    expect(nonVersionedNodeArgv).toEqual(["node", "sigil", "node-dev", "sigil", "status"]);

    const directArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["sigil", "status"],
    });
    expect(directArgv).toEqual(["node", "sigil", "status"]);

    const bunArgv = buildParseArgv({
      programName: "sigil",
      rawArgs: ["bun", "src/entry.ts", "status"],
    });
    expect(bunArgv).toEqual(["bun", "src/entry.ts", "status"]);
  });

  it("builds parse argv from fallback args", () => {
    const fallbackArgv = buildParseArgv({
      programName: "sigil",
      fallbackArgv: ["status"],
    });
    expect(fallbackArgv).toEqual(["node", "sigil", "status"]);
  });

  it("decides when to migrate state", () => {
    expect(shouldMigrateState(["node", "sigil", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "sigil", "health"])).toBe(false);
    expect(shouldMigrateState(["node", "sigil", "sessions"])).toBe(false);
    expect(shouldMigrateState(["node", "sigil", "memory", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "sigil", "agent", "--message", "hi"])).toBe(false);
    expect(shouldMigrateState(["node", "sigil", "agents", "list"])).toBe(true);
    expect(shouldMigrateState(["node", "sigil", "message", "send"])).toBe(true);
  });

  it("reuses command path for migrate state decisions", () => {
    expect(shouldMigrateStateFromPath(["status"])).toBe(false);
    expect(shouldMigrateStateFromPath(["agents", "list"])).toBe(true);
  });
});
