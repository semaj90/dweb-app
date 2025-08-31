<script lang="ts">
  import { page } from "$app/state";
  import Avatar from "$lib/components/Avatar.svelte";
  import { avatarStore } from "$lib/stores/avatarStore";
  import { onMount } from "svelte";

  let user = $state(page.data.user);
  let userStats = $state(page.data.userStats);
  let profileForm = $state({
    name: "",
    email: "",
    firstName: "",
    lastName: "",
  });

  let isUpdating = $state(false);
  let updateMessage = $state("");

  onMount(() => {
    if (user) {
      profileForm = {
        name: user?.name || "",
        email: user?.email || "",
        firstName: "", // Not available in SessionUser
        lastName: "", // Not available in SessionUser
      };
    }
    // Load avatar
    avatarStore.loadAvatar();
  });

  async function updateProfile() {
    isUpdating = true;
    updateMessage = "";

    try {
      const response = await fetch("/api/user/profile", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(profileForm),
      });

      const data = await response.json();

      if (response.ok) {
        updateMessage = "Profile updated successfully!";
        user = data.user;
      } else {
        updateMessage = data.error || "Update failed";
      }
    } catch (error) {
      updateMessage = "Network error occurred";
    } finally {
      isUpdating = false;
    }
  }
</script>

<svelte:head>
  <title>Profile Settings - WardenNet</title>
</svelte:head>

{#if user}
  <div class="space-y-4">
    <div class="space-y-4">
      <h1>Profile Settings</h1>
      <p>Manage your account information and avatar</p>
    </div>

    <div class="space-y-4">
      <!-- Avatar Section -->
      <div class="space-y-4">
        <h2>Profile Picture</h2>
        <div class="space-y-4">
          <Avatar size="large" showUploadButton={true} />
          <div class="space-y-4">
            <h3>Your Avatar</h3>
            <p>
              Upload a profile picture to personalize your account. Supported
              formats: JPEG, PNG, GIF, SVG, WebP (max 5MB)
            </p>

            {#if $avatarStore.error}
              <div class="space-y-4">
                {$avatarStore.error}
              </div>
            {/if}
          </div>
        </div>
      </div>

      <div class="space-y-4"></div>

      <!-- Profile Information -->
      <div class="space-y-4">
        <h2>Account Information</h2>

        <form submit={(e) => { e.preventDefault(); updateProfile(); }} class="space-y-4">
          <div class="space-y-4">
            <div class="space-y-4">
              <label for="name">Full Name</label>
              <input
                id="name"
                type="text"
                bind:value={profileForm.name}
                placeholder="Enter your full name"
                required
              />
            </div>

            <div class="space-y-4">
              <label for="email">Email Address</label>
              <input
                id="email"
                type="email"
                bind:value={profileForm.email}
                placeholder="Enter your email"
                required
              />
            </div>

            <div class="space-y-4">
              <label for="firstName">First Name</label>
              <input
                id="firstName"
                type="text"
                bind:value={profileForm.firstName}
                placeholder="Enter your first name"
              />
            </div>

            <div class="space-y-4">
              <label for="lastName">Last Name</label>
              <input
                id="lastName"
                type="text"
                bind:value={profileForm.lastName}
                placeholder="Enter your last name"
              />
            </div>
          </div>

          {#if updateMessage}
            <div
              class="space-y-4"
              class:success={updateMessage.includes("success")}
              class:error={!updateMessage.includes("success")}
            >
              {updateMessage}
            </div>
          {/if}

          <div class="space-y-4">
            <button type="submit" class="space-y-4" disabled={isUpdating}>
              {isUpdating ? "Updating..." : "Save Changes"}
            </button>

            <a href="/dashboard" class="space-y-4"> Cancel </a>
          </div>
        </form>
      </div>

      <div class="space-y-4"></div>

      <!-- Account Stats -->
      <div class="space-y-4">
        <h2>Account Statistics</h2>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{userStats?.totalCases || 0}</div>
            <div class="stat-label">Total Cases</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{userStats?.openCases || 0}</div>
            <div class="stat-label">Open Cases</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{userStats?.closedCases || 0}</div>
            <div class="stat-label">Closed Cases</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{userStats?.totalEvidence || 0}</div>
            <div class="stat-label">Evidence Files</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{userStats?.totalCriminals || 0}</div>
            <div class="stat-label">Persons of Interest</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{user?.role || "User"}</div>
            <div class="stat-label">Role</div>
          </div>
        </div>
      </div>
    </div>
  </div>
{:else}
  <div class="space-y-4">
    <span>Please log in to view your profile.</span>
  </div>
{/if}

<style>
  /* @unocss-include */
  .profile-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 24px;
}
  .profile-header {
    text-align: center;
    margin-bottom: 32px;
}
  .profile-header h1 {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary, #111827);
    margin-bottom: 8px;
}
  .profile-header p {
    color: var(--text-secondary, #6b7280);
    font-size: 16px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 16px;
  }

  .stat-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }

  .stat-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary, #111827);
    margin-bottom: 4px;
  }

  .stat-label {
    font-size: 14px;
    color: var(--text-secondary, #6b7280);
    font-weight: 500;
  }
  .alert {
    background: #fef3cd;
    border: 1px solid #facc15;
    color: #a16207;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    margin: 32px auto;
    max-width: 400px;
}
  /* Responsive */
  @media (max-width: 768px) {
    .form-grid {
      grid-template-columns: 1fr;
}
    .avatar-display {
      flex-direction: column;
      text-align: center;
}
}
</style>
