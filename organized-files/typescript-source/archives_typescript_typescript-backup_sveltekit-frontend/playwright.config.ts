import { defineConfig, devices } from "@playwright/test";

/**
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: "./tests",
  /* Run tests in files in parallel */
  fullyParallel: true, // Enable parallel execution for better performance
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 1,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 2 : 4, // Optimize worker count for CI vs local
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }]
  ],
  /* Global timeout */
  timeout: 2 * 60 * 1000,
  /* Expect timeout */
  expect: {
    timeout: 15_000,
    // Visual comparison settings
    threshold: 0.2,
    maxDiffPixels: 1000
  },
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: "http://localhost:5173",

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: "retain-on-failure",

    /* Screenshot settings */
    screenshot: "only-on-failure",
    video: "retain-on-failure",

    /* Viewport size */
    viewport: { width: 1440, height: 900 },

    /* Action timeout */
    actionTimeout: 30_000,

    /* Navigation timeout */
    navigationTimeout: 30_000,

    /* Ignore HTTPS errors */
    ignoreHTTPSErrors: true,

    /* User agent */
    userAgent: 'Playwright Test Agent'
  },

  /* Global setup - import test helper shims */
  globalSetup: '../../playwright-global-setup.ts',

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
    // Mobile testing
    {
      name: "Mobile Chrome",
      use: { ...devices["Pixel 5"] },
    },
    {
      name: "Mobile Safari",
      use: { ...devices["iPhone 12"] },
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
