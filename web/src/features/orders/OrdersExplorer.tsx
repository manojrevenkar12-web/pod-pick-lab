import { AnimatePresence, motion } from 'motion/react';
import { useMemo, useState } from 'react';
import { type Family, OCCLUSION, type Order, POLICIES, type Policy } from '../../shared/lib/results';
import { MOTION, OCCLUSION_LABEL, POLICY_STYLE, STATUS_LABEL } from '../../shared/theme/tokens';
import { Segmented } from '../../shared/ui/Segmented';
import { EmptyState } from '../../shared/ui/StateBlock';
import { Swatch } from '../../shared/ui/Swatch';
import s from './orders.module.css';

const FILTERS = ['disagree', 'collateral', 'all'] as const;
type Filter = (typeof FILTERS)[number];
const FILTER_LABEL: Record<Filter, string> = {
  disagree: 'Policies disagree',
  collateral: 'Something came out with it',
  all: 'All orders',
};
const PAGE = 12;

function outcome(o: Order): { text: string; tone: 'good' | 'bad' | 'plain' } {
  if (o.collateral > 0)
    return { text: `${o.picked ? 'Picked, ' : ''}${o.collateral} neighbour${o.collateral > 1 ? 's' : ''} out`, tone: 'bad' };
  if (o.dropped) return { text: 'Target dropped outside', tone: 'bad' };
  if (o.picked) return { text: 'Picked', tone: 'good' };
  return { text: STATUS_LABEL[o.status] ?? o.status, tone: 'plain' };
}

/** Every evaluated order, each policy's outcome side by side. */
export function OrdersExplorer({ orders, family }: { orders: Order[]; family: Family }) {
  const [filter, setFilter] = useState<Filter>('disagree');
  const [shown, setShown] = useState(PAGE);
  const grouped = useMemo(() => {
    const m = new Map<number, Partial<Record<Policy, Order>>>();
    for (const o of orders) {
      if (o.family !== family) continue;
      const g = m.get(o.seed) ?? {};
      g[o.policy] = o;
      m.set(o.seed, g);
    }
    return [...m.entries()].filter(([, g]) => POLICIES.every((p) => g[p]));
  }, [orders, family]);
  const rows = useMemo(
    () =>
      grouped.filter(([, g]) => {
        const os = POLICIES.map((p) => g[p]!);
        if (filter === 'collateral') return os.some((o) => o.collateral > 0);
        if (filter === 'disagree') return new Set(os.map((o) => outcome(o).text)).size > 1;
        return true;
      }),
    [grouped, filter],
  );

  return (
    <div className={s.wrap}>
      <div className={s.bar}>
        <Segmented
          label="Which orders to show"
          options={FILTERS}
          value={filter}
          onChange={(f) => {
            setFilter(f);
            setShown(PAGE);
          }}
          format={(f) => FILTER_LABEL[f]}
        />
        <p className={s.count} aria-live="polite">
          {rows.length} of {grouped.length} orders
        </p>
      </div>
      {grouped.length === 0 ? (
        <EmptyState title="No orders for this set">The per-order log was not generated for this set of bins.</EmptyState>
      ) : rows.length === 0 ? (
        <EmptyState title="Nothing to show">No order in this set matches the filter. Try another filter or set.</EmptyState>
      ) : (
        <>
          <ol className={s.list}>
            <AnimatePresence initial={false}>
              {rows.slice(0, shown).map(([seed, g]) => {
                const any = g['heuristic-open']!;
                return (
                  <motion.li
                    key={`${family}-${seed}`}
                    className={s.order}
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: MOTION.fast }}
                  >
                    <div className={s.head}>
                      <span className={`num ${s.seed}`}>Bin {seed}</span>
                      <span className={s.meta}>
                        {any.material}, {OCCLUSION_LABEL[OCCLUSION[any.occlusion_true]!].toLowerCase()}
                      </span>
                    </div>
                    <ul className={s.outcomes}>
                      {POLICIES.map((p) => {
                        const o = g[p]!;
                        const out = outcome(o);
                        return (
                          <li key={p} className={s[out.tone]}>
                            <Swatch policy={p} size={10} />
                            <span className="visually-hidden">{POLICY_STYLE[p].label}: </span>
                            <span>{out.text}</span>
                          </li>
                        );
                      })}
                    </ul>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ol>
          {shown < rows.length && (
            <button type="button" className={s.more} onClick={() => setShown((n) => n + PAGE)}>
              Show {Math.min(PAGE, rows.length - shown)} more
            </button>
          )}
        </>
      )}
    </div>
  );
}
