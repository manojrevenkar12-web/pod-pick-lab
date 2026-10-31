import { useAnimate, useReducedMotion } from 'motion/react';
import { useCallback, useEffect, useState } from 'react';
import { MOTION } from '../../shared/theme/tokens';
import styles from './hero.module.css';

// A bin seen from the side: the open front is on the left, the robot reaches in from there.
const VIEW = { w: 480, h: 270 } as const;
const BIN = { front: 150, back: 440, floor: 212, ceiling: 36, lip: 8 } as const;
const PULL = 74; // how far the target comes out before the monitor stops it
const SIGNALS = [0, 0, 1, 2, 6, 15, 23] as const; // neighbour pixels moving toward the camera, per frame
const ABORT_AT = 14;
const BAR = { x0: 170, w: 24, gap: 10, base: 258, scale: 1.4 } as const;

/** The problem in one moment: the target comes out, the item resting on it starts to ride along,
 *  the monitor sees it two frames running, and the pick is put back instead of pulling it out. */
export function BinScene() {
  const reduce = useReducedMotion();
  const [scope, animate] = useAnimate();
  const [done, setDone] = useState(false);
  const [run, setRun] = useState(0);

  const play = useCallback(async () => {
    setDone(false);
    const e = MOTION.ease;
    await Promise.all([
      animate('[data-part=arm]', { x: -170 }, { duration: 0 }),
      animate('[data-part=load]', { x: 0, rotate: 0 }, { duration: 0 }),
      animate('[data-part=rider]', { x: 0, rotate: 0 }, { duration: 0 }),
      animate('[data-part=bar]', { scaleY: 0, opacity: 0 }, { duration: 0 }),
      animate('[data-part=abort]', { opacity: 0 }, { duration: 0 }),
      animate('[data-part=cup]', { fill: 'var(--plate)' }, { duration: 0 }),
    ]);
    await animate('[data-part=arm]', { x: 0 }, { duration: MOTION.slow, ease: e });
    await animate('[data-part=cup]', { fill: 'var(--learned-fill)' }, { duration: MOTION.fast });
    const pull = { duration: 1.3, ease: 'linear' as const };
    animate('[data-part=arm]', { x: -PULL }, pull);
    animate('[data-part=load]', { x: -PULL }, pull);
    animate('[data-part=rider]', { x: -PULL * 0.92, rotate: -4 }, pull);
    for (let i = 0; i < SIGNALS.length; i++) {
      await animate(`[data-bar="${i}"]`, { scaleY: 1, opacity: 1 }, { duration: 1.3 / SIGNALS.length, ease: e });
    }
    await animate('[data-part=abort]', { opacity: 1 }, { duration: MOTION.fast });
    const back = { duration: MOTION.slow, ease: e };
    await Promise.all([
      animate('[data-part=arm]', { x: 0 }, back),
      animate('[data-part=load]', { x: 0 }, back),
      animate('[data-part=rider]', { x: 0, rotate: 0 }, back),
    ]);
    await animate('[data-part=cup]', { fill: 'var(--plate)' }, { duration: MOTION.fast });
    await animate('[data-part=arm]', { x: -170 }, { duration: MOTION.slow, ease: e });
    setDone(true);
  }, [animate]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `run` replays the sequence
  useEffect(() => {
    if (reduce) {
      setDone(true);
      return;
    }
    play();
  }, [reduce, run, play]);

  const label =
    'Side view of a bin. The robot pulls the ordered item out; the item resting on it starts to come too. ' +
    'The monitor sees neighbours moving toward the camera on two frames in a row, so the robot puts the item back and hands the order to a person.';

  return (
    <figure className={styles.scene}>
      <svg ref={scope} viewBox={`0 0 ${VIEW.w} ${VIEW.h}`} role="img" aria-label={label}>
        {/* bin */}
        <path
          d={`M${BIN.front} ${BIN.ceiling} H${BIN.back} V${BIN.floor} H${BIN.front - BIN.lip}`}
          className={styles.bin}
          fill="none"
        />
        <rect x={BIN.front - BIN.lip} y={BIN.floor - 12} width={BIN.lip} height={12} className={styles.lip} />
        <circle cx={BIN.front - 4} cy={118} r={4} className={styles.band} />
        {/* items that stay */}
        <rect x={286} y={128} width={150} height={84} rx={3} className={styles.item} />
        <rect x={300} y={92} width={92} height={36} rx={3} className={styles.item} />
        {/* the target and the item resting on it */}
        <g data-part="load">
          <rect x={158} y={150} width={104} height={62} rx={3} className={styles.target} />
        </g>
        <g data-part="rider" style={{ transformOrigin: '210px 150px' }}>
          <rect x={172} y={118} width={80} height={32} rx={3} className={styles.rider} />
        </g>
        {/* robot */}
        <g data-part="arm">
          <rect x={0} y={174} width={140} height={14} rx={7} className={styles.arm} />
          <rect data-part="cup" x={140} y={168} width={18} height={26} rx={4} className={styles.cup} />
        </g>
        {/* monitor */}
        <line
          x1={BAR.x0 - 6}
          x2={BAR.x0 + SIGNALS.length * (BAR.w + BAR.gap)}
          y1={BAR.base - ABORT_AT * BAR.scale}
          y2={BAR.base - ABORT_AT * BAR.scale}
          className={styles.threshold}
        />
        {SIGNALS.map((s, i) => (
          <rect
            // biome-ignore lint/suspicious/noArrayIndexKey: fixed sequence of frames
            key={i}
            data-part="bar"
            data-bar={i}
            x={BAR.x0 + i * (BAR.w + BAR.gap)}
            y={BAR.base - Math.max(s, 0.6) * BAR.scale}
            width={BAR.w}
            height={Math.max(s, 0.6) * BAR.scale}
            className={s >= ABORT_AT ? styles.barHot : styles.bar}
            style={{ transformOrigin: `0 ${BAR.base}px`, transformBox: 'view-box' }}
          />
        ))}
        <text
          data-part="abort"
          x={BAR.x0 + SIGNALS.length * (BAR.w + BAR.gap) + 6}
          y={BAR.base - ABORT_AT * BAR.scale + 4}
          className={styles.abortText}
        >
          abort, put back
        </text>
      </svg>
      <figcaption className={styles.caption}>
        <span>
          The item on top starts to ride out with the target. Two monitor frames over the line, and the robot puts the target
          back: the order goes to a person, nothing goes on the floor.
        </span>
        {!reduce && (
          <button type="button" className={styles.replay} onClick={() => setRun((r) => r + 1)} disabled={!done}>
            Replay
          </button>
        )}
      </figcaption>
    </figure>
  );
}
