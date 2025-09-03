<script lang="ts">
  interface Props {
    onsuccess?: () => void;
    open?: boolean;
  }

  let { onsuccess, open = true }: Props = $props();

  // Bits UI dialog primitives
  import * as Dialog from 'bits-ui';

  // State (Svelte 5 runes)
  let email = $state("");
  let password = $state("");
  let confirmPassword = $state("");
  let loading = $state(false);
  let error = $state("");

  async function handleRegister() {
    loading = true;
    error = "";
    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, confirmPassword })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Registration failed");
      onsuccess?.();
      open = false;
    } catch (e) {
      error = e instanceof Error ? e.message : "Registration failed";
    } finally {
      loading = false;
    }
  }
</script>

<Dialog.Root bind:open={open}>
  <!-- Optional external trigger could go here -->
  <Dialog.Portal>
    <Dialog.Overlay class="overlay" />
    <Dialog.Content class="content">
      <Dialog.Title class="title">Register</Dialog.Title>
      {#if error}<div class="error">{error}</div>{/if}
      <form on:submit|preventDefault={handleRegister} class="form">
        <label>
          <span>Email</span>
          <input
            type="email"
            bind:value={email}
            required
            autocomplete="email"
          />
        </label>
        <label>
          <span>Password</span>
            <input
              type="password"
              bind:value={password}
              required
              autocomplete="new-password"
            />
        </label>
        <label>
          <span>Confirm Password</span>
          <input
            type="password"
            bind:value={confirmPassword}
            required
            autocomplete="new-password"
          />
        </label>
        <div class="actions">
          <button type="submit" disabled={loading}>
            {loading ? "Registering..." : "Register"}
          </button>
          <Dialog.Close class="close-btn" type="button">Cancel</Dialog.Close>
        </div>
      </form>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: hsl(0 0% 0% / 0.5);
  }
  .content {
    position: fixed;
    top: 50%;
    left: 50%;
    width: 100%;
    max-width: 26rem;
    transform: translate(-50%, -50%);
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 10px 40px -10px hsl(0 0% 0% / 0.3);
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .title {
    font-size: 1.25rem;
    font-weight: 600;
  }
  .error {
    color: #b00020;
    font-size: 0.875rem;
  }
  form.form {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.85rem;
    font-weight: 500;
  }
  input {
    padding: 0.55rem 0.7rem;
    font-size: 0.95rem;
    border: 1px solid #d0d0d5;
    border-radius: 0.45rem;
  }
  input:focus {
    outline: 2px solid #6366f1;
    outline-offset: 1px;
  }
  .actions {
    margin-top: 0.5rem;
    display: flex;
    gap: 0.75rem;
    justify-content: flex-end;
  }
  button,
  .close-btn {
    appearance: none;
    border: none;
    border-radius: 0.45rem;
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
    cursor: pointer;
  }
  button[type="submit"] {
    background: #6366f1;
    color: white;
  }
  button[disabled] {
    opacity: 0.6;
    cursor: progress;
  }
  .close-btn {
    background: #e5e7eb;
  }
  .close-btn:hover {
    background: #d1d5db;
  }
</style>
