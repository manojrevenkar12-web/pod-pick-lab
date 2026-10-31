import { useCallback, useEffect, useState } from 'react';
import { isOrders, isResults, type Order, type Results } from './results';

export type Load =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'empty' }
  | { status: 'ready'; results: Results; orders: Order[] };

async function getJson(url: string, signal: AbortSignal): Promise<{ missing: true } | { missing: false; body: unknown }> {
  const r = await fetch(url, { signal });
  if (r.status === 404) return { missing: true };
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  return { missing: false, body: await r.json() };
}

/** Load the generated results and orders, with loading, error (retry) and empty states. */
export function useData(base = '.'): Load & { retry: () => void } {
  const [state, setState] = useState<Load>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `attempt` is the retry trigger
  useEffect(() => {
    const ctl = new AbortController();
    setState({ status: 'loading' });
    Promise.all([getJson(`${base}/results.json`, ctl.signal), getJson(`${base}/orders.json`, ctl.signal)])
      .then(([res, ord]) => {
        if (res.missing) return setState({ status: 'empty' });
        if (!isResults(res.body))
          return setState({ status: 'error', message: 'The results file is not in the expected format.' });
        const orders = !ord.missing && isOrders(ord.body) ? ord.body : [];
        setState({ status: 'ready', results: res.body, orders });
      })
      .catch((e: Error) => {
        if (e.name !== 'AbortError')
          setState({
            status: 'error',
            message: `The results could not be loaded (${e.message}). Check the connection and try again.`,
          });
      });
    return () => ctl.abort();
  }, [base, attempt]);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { ...state, retry };
}
