import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('every section renders from the generated results, without console errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('not its neighbours');
  for (const name of ['Four pick cells, the same orders', 'Who gets attempted', 'Labels or pretraining', 'Finding the ordered item', 'The monitor, in C++', 'Every order', 'Release gate']) {
    await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('switching the set of bins redraws the comparison', async ({ page }) => {
  await page.goto('/');
  const fig = page.getByRole('img', { name: /Picked, difference/ });
  const before = await fig.getAttribute('aria-label');
  await page.getByRole('button', { name: 'Dense bins' }).click();
  await expect(fig).not.toHaveAttribute('aria-label', before!);
});

test('the order filter and paging work', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'All orders' }).click();
  const items = page.locator('#orders ol > li');
  const n = await items.count();
  expect(n).toBeGreaterThan(0);
  const more = page.getByRole('button', { name: /Show \d+ more/ });
  if (await more.isVisible()) {
    await more.click();
    await expect.poll(() => items.count()).toBeGreaterThan(n);
  }
});

test('nothing scrolls sideways on a phone', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 2, name: 'Release gate' })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test('no serious accessibility violations', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 2, name: 'Release gate' })).toBeVisible();
  const a = await new AxeBuilder({ page } as unknown as ConstructorParameters<typeof AxeBuilder>[0]).withTags(['wcag2a', 'wcag2aa']).analyze();
  expect(a.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')).toEqual([]);
});
