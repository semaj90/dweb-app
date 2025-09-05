@echo off
title Fix SvelteKit TypeScript
cd sveltekit-frontend
echo Running SvelteKit sync...
npm run sync
echo Starting dev server...
npm run dev
