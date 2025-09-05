<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { createDialog, melt } from "@melt-ui/svelte";
  import { createEventDispatcher } from "svelte";
  import { writable } from "svelte/store";
  import { superForm } from "sveltekit-superforms";

  export let data: unknown;
  export let open = false;

  const dispatch = createEventDispatcher();

  const { form, errors, submitting, message, enhance } = superForm(
    data.loginForm,
    {
      onUpdated: ({ form }) => {
        if (form.valid) dispatch("success");
      },
    }
  );

  // FIX 1: The 'open' property for createDialog must be a writable store.
  // We create an internal store and sync it with the 'open' prop.
  const openStore = writable(open);

  const {
    elements: { trigger, overlay, content, title, close },
    states: { open: isMeltOpen },
  } = createDialog({
    open: openStore,
    onOpenChange: (state) => {
      open = state.next;
      if (!open) {
        dispatch("close");
      }
      return state.next;
    },
  });

  // This reactive statement ensures that if the parent changes
  // the 'open' prop, our internal store is updated.
  $: if ($openStore !== open) {
    $openStore = open;
  }
</script>

<button use:melt={$trigger} style="display: none;">Open Modal</button>

{#if $isMeltOpen}
  <div use:melt={$overlay} class="mx-auto px-4 max-w-7xl"></div>
  <div use:melt={$content} class="mx-auto px-4 max-w-7xl">
    <h2 use:melt={$title} class="mx-auto px-4 max-w-7xl">Login</h2>
    <button use:melt={$close} class="mx-auto px-4 max-w-7xl">X</button>

    <form class="mx-auto px-4 max-w-7xl" method="POST" action="?/login" use:enhance>
      <div class="mx-auto px-4 max-w-7xl">
        <label for="email" class="mx-auto px-4 max-w-7xl">Email</label>
        <input
          type="email"
          name="email"
          id="email"
          bind:value={$form.email}
          placeholder="Email"
          required
          aria-invalid={$errors.email ? "true" : undefined}
          class="mx-auto px-4 max-w-7xl"
        />
        {#if $errors.email}<span class="mx-auto px-4 max-w-7xl"
            >{$errors.email}</span
          >{/if}

        <label for="password" class="mx-auto px-4 max-w-7xl">Password</label>
        <input
          type="password"
          name="password"
          id="password"
          bind:value={$form.password}
          placeholder="Password"
          required
          aria-invalid={$errors.password ? "true" : undefined}
          class="mx-auto px-4 max-w-7xl"
        />
        {#if $errors.password}<span class="mx-auto px-4 max-w-7xl"
            >{$errors.password}</span
          >{/if}

        {#if message}<div class="mx-auto px-4 max-w-7xl">{message}</div>{/if}
      </div>
      <div class="mx-auto px-4 max-w-7xl">
        <Button
          type="button"
          variant="ghost"
          onclick={() => ($isMeltOpen = false)}
        >
          Cancel
        </Button>
        <Button type="submit" disabled={$submitting}>
          {#if $submitting}Logging in...{:else}Login{/if}
        </Button>
      </div>
    </form>
  </div>
{/if}

<style>
  .modal-content {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 50;
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18);
    min-width: 340px;
    max-width: 90vw;
    padding: 2rem 2.5rem 2rem 2.5rem;
    border: 1px solid #e5e7eb;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    animation: modal-in 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
  @keyframes modal-in {
    from {
      opacity: 0;
      transform: translate(-50%, -60%) scale(0.98);
    }
    to {
      opacity: 1;
      transform: translate(-50%, -50%) scale(1);
    }
  }
</style>
