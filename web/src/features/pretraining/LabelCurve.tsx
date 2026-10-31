import type { RiskRow } from '../../shared/lib/results';
import { CHART } from '../../shared/theme/tokens';
import s from './pretraining.module.css';

type Key = 'success_auc' | 'collateral_auc';

function mean(xs: number[]) {
  return xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);
}

/** AUC on validation orders against the number of labelled attempts, with and without pretraining. */
export function LabelCurve({ rows, metric, title }: { rows: RiskRow[]; metric: Key; title: string }) {
  const budgets = [...new Set(rows.map((r) => r.budget))].sort((a, b) => a - b);
  const series = (['scratch', 'pretrained'] as const).map((v) => ({
    v,
    pts: budgets.map((b) => {
      const xs = rows.filter((r) => r.budget === b && r.variant === v && r[metric] != null).map((r) => r[metric] as number);
      return { b, m: mean(xs), lo: Math.min(...xs), hi: Math.max(...xs), n: xs.length };
    }),
  }));
  const all = series.flatMap((x) => x.pts.filter((p) => p.n > 0).flatMap((p) => [p.lo, p.hi]));
  if (budgets.length === 0 || all.length === 0) return <p className={s.none}>No validation results for this head.</p>;
  const yMin = Math.max(0, Math.min(...all) - 0.03);
  const yMax = Math.min(1, Math.max(...all) + 0.03);
  const { curveWidth: W, curveHeight: H, pad: P } = CHART;
  const lx = (b: number) => Math.log10(b);
  const x0 = lx(budgets[0]!);
  const x1 = lx(budgets[budgets.length - 1]!) + 1e-9;
  const x = (b: number) => P * 2 + ((lx(b) - x0) / (x1 - x0)) * (W - P * 3);
  const y = (v: number) => H - P - ((v - yMin) / (yMax - yMin)) * (H - P * 2);
  const desc = series.map((sr) => `${sr.v}: ${sr.pts.map((p) => `${p.b} labels ${p.m.toFixed(3)}`).join(', ')}`).join('. ');
  return (
    <figure className={s.fig}>
      <figcaption className={s.title}>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. ${desc}`}>
        {[yMin, (yMin + yMax) / 2, yMax].map((v) => (
          <g key={v}>
            <line x1={P * 2} x2={W - P} y1={y(v)} y2={y(v)} className={s.grid} />
            <text x={P * 2 - 6} y={y(v) + 4} textAnchor="end" className={s.tick}>
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {budgets.map((b) => (
          <text key={b} x={x(b)} y={H - P + 18} textAnchor="middle" className={s.tick}>
            {b.toLocaleString('en')}
          </text>
        ))}
        {series.map((sr) => {
          const cls = sr.v === 'pretrained' ? s.pre : s.scratch;
          return (
            <g key={sr.v} className={cls}>
              {sr.pts.map((p) => (
                <line key={p.b} x1={x(p.b)} x2={x(p.b)} y1={y(p.lo)} y2={y(p.hi)} className={s.range} />
              ))}
              <polyline points={sr.pts.map((p) => `${x(p.b)},${y(p.m)}`).join(' ')} className={s.line} />
              {sr.pts.map((p) => (
                <circle key={p.b} cx={x(p.b)} cy={y(p.m)} r={5} className={s.dot} />
              ))}
            </g>
          );
        })}
      </svg>
      <p className={s.legend}>
        <span className={s.keyPre}>Pretrained on unlabelled crops</span>
        <span className={s.keyScratch}>From scratch</span>
        <span>Labelled attempts on a log scale; dots are means over seeds, lines their range.</span>
      </p>
    </figure>
  );
}
