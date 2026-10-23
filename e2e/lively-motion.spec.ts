import { expect, test } from '@playwright/test';
import { dismissGuide, samplePresent, sampleSkipReason } from './sample';

test('preview switches motion style without restarting and remembers the choice', async ({ page }) => {
  test.skip(!samplePresent, sampleSkipReason);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/'); await dismissGuide(page);
  await expect(page.getByTestId('preview-status')).toHaveAttribute('data-state', 'ready');
  const toggle = page.getByTestId('lively-toggle'), canvas = page.getByTestId('preview');
  const revision = await canvas.getAttribute('data-revision');
  await expect(toggle).not.toBeChecked();
  await toggle.check();
  await expect(toggle).toBeChecked();
  await expect(canvas).toHaveAttribute('data-revision', revision!);
  await page.reload();
  await expect(page.getByTestId('preview-status')).toHaveAttribute('data-state', 'ready');
  await expect(toggle).toBeChecked();
  await toggle.uncheck();
  await expect(toggle).not.toBeChecked();
  expect(errors).toEqual([]);
});
