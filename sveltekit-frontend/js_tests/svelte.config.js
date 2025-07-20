import adapter from "@sveltejs/adapter-auto";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://kit.svelte.dev/docs/integrations#preprocessors
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter(),

    alias: {
      $lib: "src/lib",
      $components: "src/lib/components",
      $types: "src/lib/types",
    },

    files: {
      assets: "static",
      hooks: {
        client: "src/hooks.client.ts",
        server: "src/hooks.server.ts",
      },
      lib: "src/lib",
      params: "src/params",
      routes: "src/routes",
      serviceWorker: "src/service-worker.ts",
      appTemplate: "src/app.html",
      errorTemplate: "src/error.html",
    },
  },

  vitePlugin: {
    inspector: {
      holdMode: true,
    },
  },
};

export default config;
