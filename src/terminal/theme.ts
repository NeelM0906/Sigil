import chalk, { Chalk } from "chalk";

import { SIGIL_PALETTE } from "./palette.js";

const hasForceColor =
  typeof process.env.FORCE_COLOR === "string" &&
  process.env.FORCE_COLOR.trim().length > 0 &&
  process.env.FORCE_COLOR.trim() !== "0";

const baseChalk = process.env.NO_COLOR && !hasForceColor ? new Chalk({ level: 0 }) : chalk;

const hex = (value: string) => baseChalk.hex(value);

export const theme = {
  accent: hex(SIGIL_PALETTE.accent),
  accentBright: hex(SIGIL_PALETTE.accentBright),
  accentDim: hex(SIGIL_PALETTE.accentDim),
  info: hex(SIGIL_PALETTE.info),
  success: hex(SIGIL_PALETTE.success),
  warn: hex(SIGIL_PALETTE.warn),
  error: hex(SIGIL_PALETTE.error),
  muted: hex(SIGIL_PALETTE.muted),
  heading: baseChalk.bold.hex(SIGIL_PALETTE.accent),
  command: hex(SIGIL_PALETTE.accentBright),
  option: hex(SIGIL_PALETTE.warn),
} as const;

export const isRich = () => Boolean(baseChalk.level > 0);

export const colorize = (rich: boolean, color: (value: string) => string, value: string) =>
  rich ? color(value) : value;
