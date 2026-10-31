import { useState } from 'react';
import { Forest } from '../features/comparison/Forest';
import { PolicyTable } from '../features/comparison/PolicyTable';
import { BinScene } from '../features/hero/BinScene';
import { OcclusionGrid } from '../features/occlusion/OcclusionGrid';
import { OrdersExplorer } from '../features/orders/OrdersExplorer';
import { LabelCurve } from '../features/pretraining/LabelCurve';
import { Gate, Realtime } from '../features/realtime/Realtime';
import { num, pct } from '../shared/lib/format';
import { analysisFor, FAMILIES, type Family, type Results } from '../shared/lib/results';
import { useData } from '../shared/lib/useData';
import { FAMILY_LABEL } from '../shared/theme/tokens';
import { Plate, Section } from '../shared/ui/Section';
import { Segmented } from '../shared/ui/Segmented';
import { Code, EmptyState, ErrorState, Skeleton } from '../shared/ui/StateBlock';
import styles from './app.module.css';

function Header() {
  return (
    <header className={styles.hero}>
      <div className={styles.heroText}>
        <h1>Pick the item, not its neighbours.</h1>
        <p className={styles.heroLede}>
          A suction robot takes one ordered item out of a crowded fabric bin. The hard part is what comes with it: the item
          resting on top, the one wedged beside it. This study compares four pick cells on the same simulated bins: where to put
          the cup, whether to try at all, and whether to watch the bin while pulling.
        </p>
      </div>
      <BinScene />
    </header>
  );
}

function Loading() {
  return (
    <div className={styles.loading}>
      <Skeleton height={44} width="40%" label="Loading results" />
      <Skeleton height={220} />
      <Skeleton height={160} />
    </div>
  );
}

function Study({ r, orders }: { r: Results; orders: Parameters<typeof OrdersExplorer>[0]['orders'] }) {
  const available = FAMILIES.filter((f) => analysisFor(r, f));
  const [family, setFamily] = useState<Family>(available[0] ?? 'mixed');
  const an = analysisFor(r, family);
  const lc = an?.policies['learned-closed'];
  const ho = an?.policies['heuristic-open'];
  return (
    <>
      <div className={styles.switcher}>
        <Segmented label="Set of bins" options={available} value={family} onChange={setFamily} format={(f) => FAMILY_LABEL[f]} />
        <p className={styles.switcherNote}>
          {family === 'heldout'
            ? `Written blind from a one-page brief and run once (${r.heldout_first_run?.recorded_at.slice(0, 10)}). Never rerun after changes.`
            : `${an?.policies['heuristic-open']?.orders ?? 0} orders, the same for every policy, from the same start state.`}
        </p>
      </div>

      {!an ? (
        <EmptyState title="No results for this set">This set was not evaluated in the committed run.</EmptyState>
      ) : (
        <>
          <Section
            id="compare"
            title="Four pick cells, the same orders"
            lede="Free-space ranking is the baseline: rank suction spots by the room in front of them, and hand over anything with an item resting on it. The learned cell predicts, for every spot and pull, the chance of a clean pick and the chance of bringing a neighbour out, and attempts only when that is cheaper than a person. The monitor watches the bin at 10 Hz while pulling."
            answer={
              lc && ho ? (
                <>
                  With the learned risk and the monitor, <strong>{pct(lc.pick_rate)}</strong> of orders are picked with{' '}
                  <strong>{num(lc.collateral_per_100)}</strong> neighbours out per 100 orders, against {pct(ho.pick_rate)} and{' '}
                  {num(ho.collateral_per_100)} for the ranking alone.
                </>
              ) : undefined
            }
          >
            <Plate>
              <PolicyTable analysis={an} />
            </Plate>
            <Plate>
              <Forest analysis={an} />
            </Plate>
          </Section>

          <Section
            id="occlusion"
            title="Who gets attempted"
            lede="Split by what the simulator knows about the target: free, with something resting on it, or mostly hidden behind other items. The ranking hands over everything that is not free; the learned cell decides spot by spot."
          >
            <Plate>
              <OcclusionGrid analysis={an} />
            </Plate>
          </Section>
        </>
      )}

      <Section
        id="pretraining"
        title="Labels or pretraining"
        lede={`The risk model reads a 32 × 32 depth crop around each spot. Its encoder can first learn from ${r.picks.unlabelled_crops.toLocaleString('en')} crops with no outcome, by filling in masked patches. Measured on validation orders at three label budgets.`}
        answer={`Deployed: the ${r.risk.chosen} model, chosen on validation orders.`}
      >
        <div className={styles.twoUp}>
          <Plate>
            <LabelCurve rows={r.risk.rows} metric="success_auc" title="Clean pick, AUC" />
          </Plate>
          <Plate>
            <LabelCurve rows={r.risk.rows} metric="collateral_auc" title="A neighbour comes out, AUC" />
          </Plate>
        </div>
      </Section>

      <Section
        id="perception"
        title="Finding the ordered item"
        lede="A small U-Net reads depth and intensity and is told what the order is (catalogue size, colour, weight). It returns the item's mask and whether the item is free, below others or hidden."
      >
        <Plate>
          <dl className={styles.facts}>
            <div>
              <dt>Mask overlap (IoU), validation bins</dt>
              <dd className="num">{r.perception.mask_iou_mean.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Item found</dt>
              <dd className="num">{pct(r.perception.target_found)}</dd>
            </div>
            <div>
              <dt>Occlusion class right</dt>
              <dd className="num">{pct(r.perception.occlusion_accuracy)}</dd>
            </div>
          </dl>
        </Plate>
      </Section>

      <Section
        id="realtime"
        title="The monitor, in C++"
        lede="The monitor sits in the motion loop, so it runs as a dependency-free C++ library. A Python reference implements the same arithmetic; both are run on every frame of real extractions and must agree exactly."
      >
        <Plate>
          <Realtime native={r.native} />
        </Plate>
      </Section>

      <Section
        id="orders"
        title="Every order"
        lede={`Each bin in ${FAMILY_LABEL[family].toLowerCase()}, with what each policy did.`}
      >
        <OrdersExplorer orders={orders} family={family} />
      </Section>

      <Section
        id="gate"
        title="Release gate"
        lede="Floors fail the run; targets are reported whether met or missed. Neither is lowered to pass."
      >
        <Plate>
          <Gate gate={r.gate} />
        </Plate>
      </Section>
    </>
  );
}

export function App() {
  const d = useData();
  return (
    <main className={styles.page}>
      <Header />
      {d.status === 'loading' && <Loading />}
      {d.status === 'error' && <ErrorState message={d.message} onRetry={d.retry} />}
      {d.status === 'empty' && (
        <EmptyState title="No results yet">
          Run <Code>podpick all</Code> to generate <Code>reports/results.json</Code>, then <Code>npm run sync</Code>.
        </EmptyState>
      )}
      {d.status === 'ready' && <Study r={d.results} orders={d.orders} />}
      <footer className={styles.footer}>
        <p>
          Simulated bins (MuJoCo, ray-cast camera). The numbers are rates in this simulator; the method, the paired protocol and
          the real-time monitor are what carry over. Generated by <Code>podpick report</Code>; this page computes nothing.
        </p>
      </footer>
    </main>
  );
}
