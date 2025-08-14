import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright Configuration for Legal AI RAG Testing
 * Tests Ollama, SvelteKit, PostgreSQL, and CUDA GPU acceleration
 */
export default defineConfig({
  // Test directory
  testDir: "./tests",

  // Run tests in files in parallel
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry on CI only
  retries: process.env.CI ? 2 : 0,

  // Opt out of parallel tests on CI
  workers: process.env.CI ? 1 : undefined,

  // Reporter to use
  reporter: [
    ["html"],
    ["json", { outputFile: "test-results/results.json" }],
    ["junit", { outputFile: "test-results/junit.xml" }],
  ],

  // Shared settings for all the projects below
  use: {
    // Base URL for tests
    baseURL: "http://localhost:5173",

    // Collect trace when retrying the failed test
    trace: "on-first-retry",

    // Screenshot on failure
    screenshot: "only-on-failure",

    // Video recording
    video: "retain-on-failure",

    // Ignore HTTPS errors
    ignoreHTTPSErrors: true,

    // Global timeout for each test
    actionTimeout: 30 * 1000,

    // Extra HTTP headers
    extraHTTPHeaders: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
  },

  // Configure projects for major browsers
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // Enable GPU acceleration for Chromium
        launchOptions: {
          args: [
            "--enable-gpu",
            "--use-gl=desktop",
            "--enable-webgl",
            "--enable-accelerated-2d-canvas",
            "--disable-web-security",
            "--disable-features=TranslateUI",
          ],
        },
      },
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

    // Microsoft Edge
    {
      name: "Microsoft Edge",
      use: {
        ...devices["Desktop Edge"],
        channel: "msedge",
        launchOptions: {
          args: ["--enable-gpu", "--use-gl=desktop"],
        },
      },
    },
  ],

  // Global setup and teardown
  globalSetup: "./tests/global-setup.ts",
  globalTeardown: "./tests/global-teardown.ts",

  // Web server configuration - disabled since dev server is already running
  // webServer: {
  //   command: "npm run dev",
  //   url: "http://localhost:5173",
  //   reuseExistingServer: true,
  // },

  // Test timeout
  timeout: 60 * 1000,

  // Expect timeout
  expect: {
    timeout: 10 * 1000,
  },

  // Output directory
  outputDir: "test-results/",

  // Test match patterns
  testMatch: ["**/*.spec.ts", "**/*.test.ts", "**/test-*.ts"],

  // Test ignore patterns
  testIgnore: ["**/node_modules/**", "**/dist/**", "**/.git/**"],
});
