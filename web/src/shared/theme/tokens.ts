// Theme constants used from TypeScript (charts, motion). Colours resolve CSS variables.
import type { Family, Policy } from '../lib/results';

export const MOTION = {
  fast: 0.18,
  base: 0.34,
  slow: 0.8,
  ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
} as const;

export const CHART = {
  barHeight: 10,
  rowHeight: 36,
  forestWidth: 520,
  forestRow: 34,
  forestLabel: 150,
  curveWidth: 520,
  curveHeight: 240,
  pad: 28,
} as const;

/** Hue = scorer, fill = monitor. Solid marks: the monitor was on. */
export const POLICY_STYLE: Record<Policy, { label: string; short: string; color: string; filled: boolean }> = {
  'heuristic-open': { label: 'Free-space ranking, monitor off', short: 'Ranking', color: 'var(--heuristic)', filled: false },
  'heuristic-closed': {
    label: 'Free-space ranking, monitor on',
    short: 'Ranking + monitor',
    color: 'var(--heuristic)',
    filled: true,
  },
  'learned-open': { label: 'Learned risk, monitor off', short: 'Learned', color: 'var(--learned)', filled: false },
  'learned-closed': { label: 'Learned risk, monitor on', short: 'Learned + monitor', color: 'var(--learned)', filled: true },
};

export const FAMILY_LABEL: Record<Family, string> = {
  mixed: 'Mixed bins',
  dense: 'Dense bins',
  soft: 'Mostly polybags',
  heldout: 'Held-out (blind)',
};

export const OCCLUSION_LABEL = {
  unobstructed: 'Unobstructed',
  below_others: 'Something rests on it',
  blocked: 'Mostly hidden',
} as const;

export const STATUS_LABEL: Record<string, string> = {
  picked: 'Picked',
  handoff: 'Handed over',
  aborted: 'Aborted, put back',
  lost: 'Lost on the way',
};
