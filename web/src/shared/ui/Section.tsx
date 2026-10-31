import type { ReactNode } from 'react';
import styles from './ui.module.css';

/** One question of the study: what was asked, the evidence, and the answer in a sentence or two. */
export function Section({
  id,
  title,
  lede,
  answer,
  children,
}: {
  id: string;
  title: string;
  lede?: ReactNode;
  answer?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={styles.section} aria-labelledby={`${id}-h`} id={id}>
      <div className={styles.sectionHead}>
        <h2 id={`${id}-h`}>{title}</h2>
        {lede && <p className={styles.lede}>{lede}</p>}
      </div>
      {children}
      {answer && <p className={styles.answer}>{answer}</p>}
    </section>
  );
}

export function Plate({ children }: { children: ReactNode }) {
  return <div className={styles.plate}>{children}</div>;
}
