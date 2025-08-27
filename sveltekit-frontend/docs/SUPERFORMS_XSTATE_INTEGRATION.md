Summary: How to wire sveltekit-superforms with XState (client-side) and common pitfalls.

1) Client wiring (recommended)

<script>
  import { superForm } from 'sveltekit-superforms/client';
  import { useMachine } from '@xstate/svelte';
  import { loginMachine } from '$lib/machines/loginMachine';

  export let form; // provided from server load
  const f = superForm(form);

  const [state, send] = useMachine(loginMachine, {
    services: {
      submitLogin: async () => {
        // delegate form submission to superforms which handles redirects and errors
        return f.submit();
      }
    }
  });
</script>

<form use:f on:submit={() => send('SUBMIT')}>
  <input name="email" bind:value={form.data.email} />
  <input type="password" name="password" bind:value={form.data.password} />
  <button type="submit" disabled={state.matches('submitting')}>Login</button>
</form>

Notes & pitfalls
- Ensure the server `load` returns the `form` instance (your `+page.server.ts` does).
- Use `use:f` (or the correct enhancer) so superforms attaches the handler; do not manually call fetch in XState.
- When debugging form failures, open Network tab and inspect the POST response and Set-Cookie headers.

Troubleshooting
- No Set-Cookie header: check server logs for the debug message added in `src/routes/login/+page.server.ts`.
- locals.user is null after login: visit `/api/debug/session` in dev to inspect `event.locals`.

