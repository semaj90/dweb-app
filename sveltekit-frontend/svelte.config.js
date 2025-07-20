import adapter from "@sveltejs/adapter-auto";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://kit.svelte.dev/docs/integrations#preprocessors
  // for more information about preprocessors
  preprocess: vitePreprocess(),

  kit: {
    // adapter-auto only supports some environments, see https://kit.svelte.dev/docs/adapter-auto for a list.
    // If your environment is not supported or you settled on a specific environment, switch out the adapter.
    // See https://kit.svelte.dev/docs/adapters for more information about adapters.
    adapter: adapter(),

    // TypeScript settings
    typescript: {
      config: (config) => {
        config.compilerOptions.strict = false;
        return config;
      },
    },

    // Files settings
    files: {
      assets: "static",
      hooks: {
        client: "src/hooks.client",
        server: "src/hooks.server",
      },
      lib: "src/lib",
      params: "src/params",
      routes: "src/routes",
      serviceWorker: "src/service-worker",
      appTemplate: "src/app.html",
      errorTemplate: "src/error.html",
    },
  },
};

export default config;
