import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";

export default defineConfig({
  plugins: [UnoCSS(), sveltekit()],
  server: {
    port: 5175,
    host: "0.0.0.0",
    cors: true,
    hmr: {
      port: 5176,
      clientPort: 5176
    },
    fs: {
      allow: ['..']
    }
  },
  preview: {
    port: 4173,
    host: "0.0.0.0",
    cors: true
  },
  build: {
    rollupOptions: {
      external: [
        "amqplib",
        "ioredis",
        "@qdrant/js-client-rest",
        "neo4j-driver",
        "@xstate/svelte",
        "xstate",
      ],
    },
  },
});
