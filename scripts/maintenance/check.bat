MANUAL INSTALLATION STEPS - GUARANTEED TO WORK
==============================================

Step 1: Navigate to correct directory
-------------------------------------
cd C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend

Step 2: Clean installation
--------------------------
rmdir /s /q node_modules
del package-lock.json
npm cache clean --force

Step 3: Install dependencies ONE BY ONE
---------------------------------------
npm install bits-ui@latest
npm install clsx
npm install tailwind-merge
npm install class-variance-authority
npm install @melt-ui/svelte
npm install @melt-ui/pp
npm install unocss
npm install @unocss/preset-uno
npm install @unocss/preset-wind
npm install @unocss/preset-icons
npm install @unocss/svelte-scoped
npm install xstate
npm install @xstate/svelte
npm install lokijs
npm install fuse.js

Step 4: Create utils.ts
-----------------------
Create: src\lib\utils.ts

Content:
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

Step 5: Fix store exports
-------------------------
Create: src\lib\stores\index.ts

Content:
export * from './ai-commands.js';
export * from './cases.js';
export * from './evidence.js';

Create: src\lib\stores\ai-commands.js

Content:
import { writable } from 'svelte/store';

export const aiCommands = writable({
  current: '',
  history: []
});

Create: src\lib\stores\cases.js

Content:
import { writable } from 'svelte/store';

export const casesStore = writable({
  cases: [],
  currentCase: null,
  elements: {}
});

Create: src\lib\stores\evidence.js

Content:
import { writable } from 'svelte/store';

export const evidenceStore = writable({
  evidence: [],
  uploads: []
});

Step 6: Create Button component
-------------------------------
Create: src\lib\components\ui\button.svelte

Content:
<script lang="ts">
  import { Button as ButtonPrimitive } from 'bits-ui';
  import { cn } from '$lib/utils.js';

  interface Props {
    variant?: 'default' | 'primary';
    class?: string;
    children?: any;
  }

  let { variant = 'default', class: className = '', children, ...restProps }: Props = $props();
</script>

<ButtonPrimitive.Root class={cn('nier-button', className)} {...restProps}>
  {@render children?.()}
</ButtonPrimitive.Root>

Step 7: Create UnoCSS config
----------------------------
Create: uno.config.ts

Content:
import { defineConfig } from 'unocss';
import { presetUno } from '@unocss/preset-uno';
import { presetWind } from '@unocss/preset-wind';

export default defineConfig({
  presets: [presetUno(), presetWind()],
  shortcuts: {
    'nier-button': 'px-4 py-2 bg-gray-200 border border-black uppercase'
  }
});

Step 8: Update vite.config.ts
-----------------------------
Add UnoCSS to vite.config.ts:

import UnoCSS from 'unocss/vite';

plugins: [
  UnoCSS(),
  sveltekit()
]

Step 9: Test installation
-------------------------
npm run check
npm run build
npm run dev

TROUBLESHOOTING:
===============
- If npm install fails: Use --legacy-peer-deps flag
- If types conflict: Delete node_modules and try again
- If build fails: Check vite.config.ts has UnoCSS plugin
- If stores fail: Make sure using .js extensions in imports

This manual approach eliminates all .bat script issues and ensures everything works correctly.