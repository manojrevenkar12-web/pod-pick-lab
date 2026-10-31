import { num, pct } from '../../shared/lib/format';
import { type Analysis, POLICIES } from '../../shared/lib/results';
import { POLICY_STYLE } from '../../shared/theme/tokens';
import { ScrollRegion } from '../../shared/ui/ScrollRegion';
import { Swatch } from '../../shared/ui/Swatch';
import t from '../../shared/ui/table.module.css';
import s from './comparison.module.css';

function Bar({ value, max, tone }: { value: number; max: number; tone: 'good' | 'bad' | 'plain' }) {
  const w = max > 0 ? Math.min(1, value / max) : 0;
  return (
    <span className={s.track} aria-hidden="true">
      <span className={s[tone]} style={{ width: `${100 * w}%` }} />
    </span>
  );
}

/** The four policies on the same orders: what came out, what came with it, what it cost. */
export function PolicyTable({ analysis }: { analysis: Analysis }) {
  const rows = POLICIES.filter((p) => analysis.policies[p]);
  const maxColl = Math.max(...rows.map((p) => analysis.policies[p]!.collateral_per_100), 1);
  const maxCost = Math.max(...rows.map((p) => analysis.policies[p]!.cost_per_order), 1);
  return (
    <ScrollRegion label="Results per policy">
      <table className={t.table}>
        <thead>
          <tr>
            <th scope="col">Policy</th>
            <th scope="col">Picked</th>
            <th scope="col">Neighbours out per 100 orders</th>
            <th scope="col" className={t.right}>
              Target dropped
            </th>
            <th scope="col" className={t.right}>
              Put back
            </th>
            <th scope="col">Cost per order</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => {
            const v = analysis.policies[p]!;
            return (
              <tr key={p}>
                <th scope="row" className={s.policy}>
                  <Swatch policy={p} />
                  {POLICY_STYLE[p].label}
                </th>
                <td>
                  <span className={s.cell}>
                    <span className={`num ${s.value}`}>{pct(v.pick_rate)}</span>
                    <Bar value={v.pick_rate} max={1} tone="good" />
                  </span>
                </td>
                <td>
                  <span className={s.cell}>
                    <span className={`num ${s.value}`}>{num(v.collateral_per_100)}</span>
                    <Bar value={v.collateral_per_100} max={maxColl} tone="bad" />
                  </span>
                </td>
                <td className={t.right}>{pct(v.drop_rate, 1)}</td>
                <td className={t.right}>{pct(v.abort_rate)}</td>
                <td>
                  <span className={s.cell}>
                    <span className={`num ${s.value}`}>{num(v.cost_per_order)} s</span>
                    <Bar value={v.cost_per_order} max={maxCost} tone="plain" />
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </ScrollRegion>
  );
}
