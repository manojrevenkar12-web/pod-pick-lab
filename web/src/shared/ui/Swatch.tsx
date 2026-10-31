import type { Policy } from '../lib/results';
import { POLICY_STYLE } from '../theme/tokens';

/** The policy mark used everywhere: hue for the scorer, solid when the monitor was on. */
export function Swatch({ policy, size = 12 }: { policy: Policy; size?: number }) {
  const s = POLICY_STYLE[policy];
  const r = size / 2 - 1;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true" style={{ flex: 'none' }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill={s.filled ? s.color : 'none'} stroke={s.color} strokeWidth={2} />
    </svg>
  );
}
