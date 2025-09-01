Svelte 5 Editor Migration — First Steps

Goal: Incrementally make the rich-text / block-editor components (TipTap-based) compatible with Svelte 5 runes-mode with minimal, auditable changes.

1) Add local type shims (done)
   - `src/lib/types/mermaid.d.ts`
   - `src/lib/types/tiptap-shims.d.ts`

2) Focused svelte-check for editor files
   - `npm run check:editor` runs svelte-check against `src/lib/components/editor/*` and `*RichTextEditor.svelte` files only.

3) Edit pattern per file (one commit per fix)
   - Replace legacy `$:` reactive labels with `$effect` or `$derived` where appropriate.
   - Merge duplicate top-level `<script>` blocks into one `<script>` and one optional `<script context="module">`.
   - Replace deprecated `on:` handlers or convert to Svelte 5 event handlers where necessary.
   - Add `// @ts-ignore` or small type shims only if necessary; prefer correct typing.

4) Slash-commands & Mermaid
   - Use TipTap's plugin system for slash commands; implement a small `slash-menu` plugin that opens a command palette when `/` is typed.
   - Lazy-load `mermaid` in the client (`await import('mermaid')`) and render diagrams server-side only if SSR is disabled.

5) Verification
   - After each file change run `npm run check:editor` and commit if green.

Recommended first file
- `src/lib/components/editor/TiptapWithAIAssistant.svelte` — highest impact, integrate AI hooks, likely causes duplicate `<script>` and reactive `$:` complaints.

If you'd like I can apply the first change (convert `$:` to `$effect` in `TiptapWithAIAssistant.svelte`) and run `npm run check:editor`; say "apply" to proceed.
