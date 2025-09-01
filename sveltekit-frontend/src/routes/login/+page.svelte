<script lang="ts">
  import { goto } from '$app/navigation';
let email = $state('');
let password = $state('');

  async function submit(e: Event) {
    e.preventDefault();
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    if (res.ok) {
      goto('/dashboard');
    } else {
      alert('Login failed');
    }
  }
</script>

<form onsubmit={submit}>
  <input bind:value={email} placeholder="Email" type="email" required />
  <input bind:value={password} placeholder="Password" type="password" required />
  <button type="submit">Login</button>
</form>
