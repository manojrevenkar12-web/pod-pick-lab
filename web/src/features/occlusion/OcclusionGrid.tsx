import { num, pct } from '../../shared/lib/format';
import { type Analysis, OCCLUSION, POLICIES, type Summary } from '../../shared/lib/results';
import { OCCLUSION_LABEL, POLICY_STYLE } from '../../shared/theme/tokens';
import { Swatch } from '../../shared/ui/Swatch';
import s from './occlusion.module.css';

const isSummary = (x: unknown): x is Summary => typeof x === 'object' && x !== null && 'pick_rate' in x;

/** Who gets attempted: per true occlusion class, how often each policy picked and what came with it. */
export function OcclusionGrid({ analysis }: { analysis: Analysis }) {
  const pols = POLICIES.filter((p) => analysis.by_occlusion[p]);
  return (
    <div className={s.grid}>
      {OCCLUSION.map((k) => {
        const first = analysis.by_occlusion[pols[0]!]?.[k];
        const n = isSummary(first) ? first.orders : 0;
        return (
          <div key={k} className={s.col}>
            <h3>
              {OCCLUSION_LABEL[k]} <span className={s.n}>{n} orders</span>
            </h3>
            {n === 0 ? (
              <p className={s.none}>No orders of this kind in this set.</p>
            ) : (
              <ul className={s.list}>
                {pols.map((p) => {
                  const v = analysis.by_occlusion[p]?.[k];
                  if (!isSummary(v)) return null;
                  return (
                    <li key={p} className={s.row}>
                      <Swatch policy={p} />
                      <span className={s.name}>{POLICY_STYLE[p].short}</span>
                      <span className={s.meter} aria-hidden="true">
                        <span style={{ width: `${100 * v.pick_rate}%` }} />
                      </span>
                      <span className={`num ${s.val}`}>{pct(v.pick_rate)}</span>
                      <span className={`num ${s.coll}`} title="neighbours out per 100 orders">
                        {num(v.collateral_per_100)}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        );
      })}
      <p className={s.key}>
        Bar and first number: share of orders picked. Second number, in red: neighbours pulled out per 100 orders.
      </p>
    </div>
  );
}
