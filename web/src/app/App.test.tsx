import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';
import fixture from './fixture.json';
import orders from './orders.fixture.json';

function respond(map: Record<string, { status: number; body?: unknown }>) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const key = Object.keys(map).find((k) => url.endsWith(k));
      const r = key ? map[key]! : { status: 404 };
      return { ok: r.status === 200, status: r.status, json: async () => r.body };
    }),
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('App', () => {
  it('shows a skeleton while loading', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));
    render(<App />);
    expect(screen.getByRole('status', { name: 'Loading results' })).toBeTruthy();
  });

  it('shows an error with a retry that loads again', async () => {
    respond({ 'results.json': { status: 500 } });
    render(<App />);
    const retry = await screen.findByRole('button', { name: 'Try again' });
    respond({ 'results.json': { status: 200, body: fixture }, 'orders.json': { status: 200, body: orders } });
    await act(async () => fireEvent.click(retry));
    await screen.findByRole('heading', { level: 2, name: 'Four pick cells, the same orders' });
  });

  it('explains what to run when there are no results', async () => {
    respond({});
    render(<App />);
    await screen.findByText('No results yet');
  });

  it('rejects a file in the wrong format', async () => {
    respond({ 'results.json': { status: 200, body: { hello: 1 } } });
    render(<App />);
    await screen.findByText(/not in the expected format/);
  });

  it('renders every section from the results and switches the set of bins', async () => {
    respond({ 'results.json': { status: 200, body: fixture }, 'orders.json': { status: 200, body: orders } });
    render(<App />);
    for (const name of ['Four pick cells, the same orders', 'Who gets attempted', 'Labels or pretraining', 'Finding the ordered item', 'The monitor, in C++', 'Every order', 'Release gate']) {
      await screen.findByRole('heading', { level: 2, name });
    }
    const dense = screen.getByRole('button', { name: 'Dense bins' });
    fireEvent.click(dense);
    await waitFor(() => expect(dense.getAttribute('aria-pressed')).toBe('true'));
  });

  it('shows an empty state when a filter matches no order', async () => {
    respond({ 'results.json': { status: 200, body: fixture }, 'orders.json': { status: 200, body: [] } });
    render(<App />);
    await screen.findByText('No orders for this set');
  });
});
