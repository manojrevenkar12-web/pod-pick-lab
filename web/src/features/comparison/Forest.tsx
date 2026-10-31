import { signed } from '../../shared/lib/format';
import { type Analysis, type Diff, POLICIES, type Policy } from '../../shared/lib/results';
import { CHART, POLICY_STYLE } from '../../shared/theme/tokens';
import s from './comparison.module.css';

const METRICS = [
  { key: 'picked', title: 'Picked', scale: 100, unit: ' pts', better: 'up' },
  { key: 'collateral', title: 'Neighbours out per 100 orders', scale: 100, unit: '', better: 'down' },
  { key: 'cost', title: 'Cost per order (s)', scale: 1, unit: ' s', better: 'down' },
] as const;

function Panel({ metric, rows }: { metric: (typeof METRICS)[number]; rows: [Policy, Diff][] }) {
  const W = CHART.forestWidth - CHART.forestLabel;
  const vals = rows.flatMap(([, d]) => [d.lo, d.hi, 0].map((v) => v * metric.scale));
  const span = Math.max(...vals.map(Math.abs), 1e-9) * 1.15;
  const x = (v: number) => CHART.pad + ((v * metric.scale + span) / (2 * span)) * (W - 2 * CHART.pad);
  const H = rows.length * CHART.forestRow + CHART.pad;
  const summary = rows
    .map(
      ([p, d]) =>
        `${POLICY_STYLE[p].short}: ${signed(d.diff * metric.scale, 1, metric.unit)} (95% ${signed(d.lo * metric.scale, 1)} to ${signed(d.hi * metric.scale, 1)})`,
    )
    .join('; ');
  return (
    <figure className={s.panel}>
      <figcaption className={s.panelTitle}>
        {metric.title} <span className={s.better}>{metric.better === 'up' ? 'higher is better' : 'lower is better'}</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${metric.title}, difference from the free-space ranking without monitor. ${summary}`}
      >
        <line x1={x(0)} x2={x(0)} y1={0} y2={H - CHART.pad / 2} className={s.zero} />
        {rows.map(([p, d], i) => {
          const y = CHART.pad / 2 + i * CHART.forestRow + CHART.forestRow / 2;
          const st = POLICY_STYLE[p];
          return (
            <g key={p}>
              <line x1={x(d.lo)} x2={x(d.hi)} y1={y} y2={y} stroke={st.color} strokeWidth={3} strokeLinecap="round" />
              <circle
                cx={x(d.diff)}
                cy={y}
                r={6}
                fill={st.filled ? st.color : 'var(--plate)'}
                stroke={st.color}
                strokeWidth={2.5}
              />
              <text x={x(d.hi) + 10} y={y + 4} className={s.tick}>
                {signed(d.diff * metric.scale, 1)}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

/** Each change against the free-space ranking without the monitor, paired over the same orders, 95% bootstrap. */
export function Forest({ analysis }: { analysis: Analysis }) {
  const vs = analysis.vs_heuristic_open ?? {};
  const pols = POLICIES.filter((p) => p !== 'heuristic-open' && vs[p]);
  return (
    <div className={s.forest}>
      <ul className={s.legend} aria-label="Policies">
        {pols.map((p) => (
          <li key={p}>
            <svg width={14} height={14} aria-hidden="true">
              <circle
                cx={7}
                cy={7}
                r={5}
                fill={POLICY_STYLE[p].filled ? POLICY_STYLE[p].color : 'none'}
                stroke={POLICY_STYLE[p].color}
                strokeWidth={2}
              />
            </svg>
            {POLICY_STYLE[p].label}
          </li>
        ))}
      </ul>
      <div className={s.panels}>
        {METRICS.map((m) => (
          <Panel key={m.key} metric={m} rows={pols.map((p) => [p, vs[p]![m.key]] as [Policy, Diff])} />
        ))}
      </div>
    </div>
  );
}
