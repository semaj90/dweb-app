import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { context7CompatibleLibraryID, topic, tokens = 10000 } = await request.json();
    
    if (!context7CompatibleLibraryID) {
      return json({ error: 'context7CompatibleLibraryID is required' }, { status: 400 });
    }

    const libraryDocs = {
      '/svelte/svelte': {
        content: '# Svelte 5 Runes\n\n- `let count = $state(0)` - reactive state\n- `let doubled = $derived(count * 2)` - computed values\n- `$effect(() => console.log(count))` - side effects',
        metadata: { library: 'svelte', topic: topic || 'runes', tokenCount: 150 },
        snippets: [{ title: 'State', code: 'let count = $state(0);', description: 'Reactive state' }]
      },
      '/melt-ui/melt-ui': {
        content: '# Melt UI v0.39.0\n\n```js\nconst { elements: { root } } = createButton();\n```\n```svelte\n<button use:melt={$root}>Click</button>\n```',
        metadata: { library: 'melt-ui', version: '0.39.0', topic: topic || 'builders', tokenCount: 120 },
        snippets: [{ title: 'Button', code: 'const { elements: { root } } = createButton();', description: 'Melt button builder' }]
      },
      '/bits-ui/bits-ui': {
        content: '# Bits UI v2\n\n```svelte\n<Dialog.Root bind:open={isOpen}>\n  <Dialog.Trigger>Open</Dialog.Trigger>\n  <Dialog.Content>\n    <Dialog.Title>Title</Dialog.Title>\n  </Dialog.Content>\n</Dialog.Root>\n```',
        metadata: { library: 'bits-ui', version: '2.x', topic: topic || 'dialog', tokenCount: 140 }
      },
      '/xstate/xstate': {
        content: '# XState v5\n\n```js\nconst machine = createMachine({\n  initial: "idle",\n  states: {\n    idle: { on: { START: "active" } },\n    active: { on: { STOP: "idle" } }\n  }\n});\n```',
        metadata: { library: 'xstate', version: '5.x', topic: topic || 'machines', tokenCount: 130 }
      }
    };

    const result = libraryDocs[context7CompatibleLibraryID] || {
      content: `# ${context7CompatibleLibraryID}\n\nDocumentation not available for this library.`,
      metadata: { library: context7CompatibleLibraryID, tokenCount: 20 }
    };

    return json({ success: true, ...result, requestedTokens: tokens, timestamp: new Date().toISOString() });
    
  } catch (error) {
    return json({ success: false, error: error.message }, { status: 500 });
  }
};
