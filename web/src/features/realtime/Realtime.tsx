import { pct } from '../../shared/lib/format';
import type { Results } from '../../shared/lib/results';
import s from './realtime.module.css';

/** The C++ monitor against its Python reference, and the release gate. */
export function Realtime({ native }: { native: Results['native'] }) {
  return (
    <dl className={s.facts}>
      <div>
        <dt>Frames compared</dt>
        <dd className="num">{native.frames.toLocaleString('en')}</dd>
      </div>
      <div>
        <dt>Same signal and action as the reference</dt>
        <dd className="num">
          {pct(native.signal_equal)} / {pct(native.action_equal)}
        </dd>
      </div>
      <div>
        <dt>C++ per 64 × 48 frame (median, p99)</dt>
        <dd className="num">
          {native.native_us_median.toFixed(0)} µs, {native.native_us_p99.toFixed(0)} µs
        </dd>
      </div>
      <div>
        <dt>Python reference per frame</dt>
        <dd className="num">{native.python_us_median.toFixed(0)} µs</dd>
      </div>
    </dl>
  );
}

export function Gate({ gate }: { gate: Results['gate'] }) {
  const fmt = (v: number | null) => (v == null ? 'n/a' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
  return (
    <div className={s.gate}>
      <ul className={s.checks} aria-label="Release floors">
        {gate.floors.map((c) => (
          <li key={c.name} className={c.ok ? s.pass : s.fail}>
            <span className={s.mark}>{c.ok ? 'Pass' : 'Fail'}</span>
            <span>{c.name}</span>
            <span className={`num ${s.actual}`}>
              {fmt(c.actual)} {c.op} {c.value}
            </span>
          </li>
        ))}
      </ul>
      <ul className={s.checks} aria-label="Targets, reported but not gated">
        {gate.targets.map((c) => (
          <li key={c.name} className={c.ok ? s.met : s.missed}>
            <span className={s.mark}>{c.ok ? 'Met' : 'Missed'}</span>
            <span>{c.name}</span>
            <span className={`num ${s.actual}`}>
              {fmt(c.actual)} {c.op} {c.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
