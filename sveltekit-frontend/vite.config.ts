import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";

export default defineConfig({
  plugins: [UnoCSS(), sveltekit()],
  server: {
    port: 5173,
    host: "127.0.0.1",
  },
  preview: {
    port: 4173,
    host: true,
  },
  build: {
    rollupOptions: {
      external: ['amqplib', 'ioredis', '@qdrant/js-client-rest', 'neo4j-driver', '@xstate/svelte', 'xstate'],
    },
  },
});
