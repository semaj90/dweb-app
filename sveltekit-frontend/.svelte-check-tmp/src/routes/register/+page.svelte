<script lang="ts">
  import { goto } from '$app/navigation';
  let email = '';
  let password = '';

  async function submit(e: Event) {
    e.preventDefault();
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    if (res.ok) {
      goto('/login');
    } else {
      alert('Registration failed');
    }
  }
</script>

<form on:submit={submit}>
  <input bind:value={email} placeholder="Email" type="email" required />
  <input bind:value={password} placeholder="Password" type="password" required />
  <button type="submit">Register</button>
</form>
