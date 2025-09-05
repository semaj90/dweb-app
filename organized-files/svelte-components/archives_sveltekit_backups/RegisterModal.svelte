<script lang="ts">
  import Modal from "../ui/Modal.svelte";
  // Adjust path to melt-ui or bits-ui as needed
  import { createEventDispatcher } from "svelte";
  let email = "";
  let password = "";
  let confirmPassword = "";
  let loading = false;
  let error = "";
  const dispatch = createEventDispatcher();

  async function handleRegister() {
    loading = true;
    error = "";
    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, confirmPassword }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Registration failed");
      dispatch("success", data);
    } catch (e) {
      error = e instanceof Error ? e.message : "Registration failed";
    } finally {
      loading = false;
    }
  }
</script>

<Modal open>
  <form onsubmit|preventDefault={handleRegister}>
    <h2>Register</h2>
    {#if error}<div class="mx-auto px-4 max-w-7xl">{error}</div>{/if}
    <input type="email" bind:value={email} placeholder="Email" required />
    <input
      type="password"
      bind:value={password}
      placeholder="Password"
      required
    />
    <input
      type="password"
      bind:value={confirmPassword}
      placeholder="Confirm Password"
      required
    />
    <button type="submit" disabled={loading}
      >{loading ? "Registering..." : "Register"}</button
    >
  </form>
</Modal>

<style>
  .error {
    color: red;
    margin-bottom: 1rem;
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  input,
  button {
    font-size: 1rem;
  }
</style>
