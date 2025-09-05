<script lang="ts">
  interface Props {
    onsuccess?: (event?: any) => void;
  }


  import Modal from "../ui/Modal.svelte";
  // Adjust path to melt-ui or bits-ui as needed
    let email = "";
  let password = "";
  let confirmPassword = "";
  let loading = false;
  let error = "";
  
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
      onsuccess?.();
    } catch (e) {
      error = e instanceof Error ? e.message : "Registration failed";
    } finally {
      loading = false;
}}
</script>

<Modal open>
  <form on:submit|preventDefault={handleRegister}>
    <h2>Register</h2>
    {#if error}<div class="space-y-4">{error}</div>{/if}
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
  /* @unocss-include */
  .error {
    color: red
    margin-bottom: 1rem;
}
  form {
    display: flex
    flex-direction: column
    gap: 1rem;
}
  input,
  button {
    font-size: 1rem;
}
</style>
