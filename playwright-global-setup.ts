// Global setup entry for Playwright to import our test helpers.
export default async function globalSetup() {
  // Import side-effectful setup (registers helpers)
  try {
    await import('./sveltekit-frontend/tests/setup/playwright-setup');
  } catch (e) {
    // If helpers fail to load, still continue; tests may import helpers explicitly.
    // Keep this quiet to avoid breaking test runs due to optional setup.
    // eslint-disable-next-line no-console
    console.warn('Playwright global setup: failed to import helpers', e);
  }
}
