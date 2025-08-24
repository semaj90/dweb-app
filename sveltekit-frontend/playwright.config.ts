import { defineConfig, devices } from "@playwright/test";

/**
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: "./tests",
  /* Run tests in files in parallel */
  fullyParallel: false, // Disable for session testing to avoid conflicts
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 1,
  /* Opt out of parallel tests on CI. */
  workers: 1, // Force single worker for session tests
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: "list",
  /* Global timeout */
  timeout: 2 * 60 * 1000,
  /* Expect timeout */
  expect: { timeout: 10_000 },
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: "http://localhost:5173",

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: "on-first-retry",

    /* Screenshot settings */
    screenshot: "only-on-failure",
    video: "retain-on-failure",

    /* Viewport size */
    viewport: { width: 1280, height: 800 },
    
    /* Action timeout */
    actionTimeout: 20_000,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  /* Start dev server before running tests */
  webServer: {
    command: "npm run dev",
    port: 5173,
    env: {
      NODE_ENV: "testing",
    },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
