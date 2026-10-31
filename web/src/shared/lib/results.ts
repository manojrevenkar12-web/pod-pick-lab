// The shape of reports/results.json and reports/orders.json, as far as the dashboard reads them.

export const POLICIES = ['heuristic-open', 'heuristic-closed', 'learned-open', 'learned-closed'] as const;
export type Policy = (typeof POLICIES)[number];
export const FAMILIES = ['mixed', 'dense', 'soft', 'heldout'] as const;
export type Family = (typeof FAMILIES)[number];
export const OCCLUSION = ['unobstructed', 'below_others', 'blocked'] as const;
export type Occlusion = (typeof OCCLUSION)[number];

export interface Summary {
  orders: number;
  pick_rate: number;
  attempt_rate: number;
  success_when_attempted: number | null;
  collateral_per_100: number;
  orders_with_collateral: number;
  drop_rate: number;
  wrong_item_rate: number;
  abort_rate: number;
  robot_s_per_order: number;
  uph: number;
  cost_per_order: number;
}

export interface Diff {
  diff: number;
  lo: number;
  hi: number;
  n: number;
}

export interface Analysis {
  policies: Partial<Record<Policy, Summary>>;
  by_occlusion: Partial<Record<Policy, Record<Occlusion, Summary | { orders: 0 }>>>;
  vs_heuristic_open?: Partial<Record<Policy, Record<'picked' | 'collateral' | 'dropped' | 'cost', Diff>>>;
  learned_promises?: {
    attempts: number;
    predicted_success: number;
    observed_success: number;
    predicted_collateral: number;
    observed_collateral: number;
  };
}

export interface RiskRow {
  budget: number;
  seed: number;
  variant: 'scratch' | 'pretrained';
  success_auc: number | null;
  collateral_auc: number | null;
}

export interface Check {
  name: string;
  op: string;
  value: number;
  actual: number | null;
  ok: boolean;
}

export interface Results {
  generated_at: string;
  profile: string;
  perception: {
    mask_iou_mean: number;
    target_found: number;
    occlusion_accuracy: number;
    occlusion_confusion: number[][];
    train_samples: number;
  };
  picks: { attempts: number; success: number; collateral: number; drop: number; unlabelled_crops: number };
  risk: {
    rows: RiskRow[];
    chosen: string;
    final: Record<'success' | 'collateral' | 'drop', { auc: number | null; ece: number }>;
  };
  native: {
    frames: number;
    signal_equal: number;
    action_equal: number;
    native_us_median: number;
    native_us_p99: number;
    python_us_median: number;
  };
  evaluation: Partial<Record<Family, { analysis: Analysis }>>;
  heldout_first_run?: { recorded_at: string; spec_sha256: string; orders: number; analysis: Analysis };
  config: { decision: Record<string, number>; monitor: Record<string, number> };
  gate: { floors: Check[]; targets: Check[] };
}

export interface Order {
  family: Family;
  seed: number;
  policy: Policy;
  occlusion_true: number;
  material: 'box' | 'bag' | 'book';
  status: 'picked' | 'handoff' | 'aborted' | 'lost';
  reason: string | null;
  attempts: number;
  robot_s: number;
  picked: boolean;
  collateral: number;
  dropped: boolean;
  wrong_item: boolean;
  first_primitive: string | null;
  first_p: [number, number, number] | null;
  max_signal: number;
}

export function isResults(x: unknown): x is Results {
  if (typeof x !== 'object' || x === null) return false;
  const r = x as Record<string, unknown>;
  return typeof r.generated_at === 'string' && typeof r.evaluation === 'object' && typeof r.gate === 'object';
}

export function isOrders(x: unknown): x is Order[] {
  return Array.isArray(x) && (x.length === 0 || (typeof x[0] === 'object' && x[0] !== null && 'policy' in x[0]));
}

/** The analysis for a family: test families from the evaluation, the blind family from its first run. */
export function analysisFor(r: Results, f: Family): Analysis | undefined {
  if (f === 'heldout') return r.heldout_first_run?.analysis;
  return r.evaluation[f]?.analysis;
}
