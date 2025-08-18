<!-- YoRHa Unit Profile Page -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { invalidateAll } from '$app/navigation';
  import type { PageData } from './$types';
  
  export let data: PageData;
  
  $: user = data.user;
  $: activities = data.activities;
  $: achievements = data.achievements;
  $: equipment = data.equipment;
  
  let activeTab = 'overview';
  let isEditing = false;
  let editData = { ...user };
  let uploadingAvatar = false;
  let avatarFile: File | null = null;
  
  // YoRHa colors
  const colors = {
    bg: '#D4D3A7',
    text: '#454138',
    accent: '#BAA68C',
    border: '#8B8680',
    highlight: '#CDC8B0',
    error: '#8B4513',
    success: '#6B7353'
  };
  
  // Calculate XP progress
  $: xpProgress = user ? (user.xp / getMaxXp(user.level)) * 100 : 0;
  
  function getMaxXp(level: number): number {
    return level * 1000 + 1000;
  }
  
  // Handle edit mode
  function startEdit() {
    isEditing = true;
    editData = { ...user };
  }
  
  function cancelEdit() {
    isEditing = false;
    editData = { ...user };
  }
  
  async function saveProfile() {
    try {
      const response = await fetch('/api/user/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: editData.name,
          bio: editData.bio,
          settings: editData.settings
        })
      });
      
      if (response.ok) {
        isEditing = false;
        await invalidateAll();
      } else {
        console.error('Failed to update profile');
      }
    } catch (error) {
      console.error('Error updating profile:', error);
    }
  }
  
  // Handle avatar upload
  async function handleAvatarChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      avatarFile = input.files[0];
      await uploadAvatar();
    }
  }
  
  async function uploadAvatar() {
    if (!avatarFile) return;
    
    uploadingAvatar = true;
    const formData = new FormData();
    formData.append('avatar', avatarFile);
    
    try {
      const response = await fetch('/api/user/avatar', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        await invalidateAll();
      }
    } catch (error) {
      console.error('Error uploading avatar:', error);
    } finally {
      uploadingAvatar = false;
      avatarFile = null;
    }
  }
  
  // Activity type icons
  const activityIcons: Record<string, string> = {
    login: '🔐',
    logout: '🚪',
    mission_start: '🎯',
    mission_complete: '✅',
    level_up: '⬆️',
    achievement_unlock: '🏆',
    equipment_change: '🛡️',
    profile_update: '✏️',
    combat_action: '⚔️',
    system_sync: '🔄'
  };
  
  // Format date
  function formatDate(date: string | Date): string {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(new Date(date));
  }
  
  // Format time ago
  function timeAgo(date: string | Date): string {
    const seconds = Math.floor((Date.now() - new Date(date).getTime()) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }
</script>

<div class="profile-container">
  <div class="profile-header" style="background-color: {colors.highlight}; border-color: {colors.border}">
    <div class="profile-header-content">
      <!-- Avatar Section -->
      <div class="avatar-section">
        <div class="avatar-container" style="border-color: {colors.border}">
          {#if user?.avatarUrl}
            <img src={user.avatarUrl} alt="Unit Avatar" class="avatar-image" />
          {:else}
            <div class="avatar-placeholder" style="background-color: {colors.accent}">
              <span class="avatar-icon">👤</span>
            </div>
          {/if}
          {#if isEditing}
            <label class="avatar-upload" style="background-color: {colors.text}">
              <input type="file" accept="image/*" on:change={handleAvatarChange} hidden />
              <span style="color: {colors.bg}">📷</span>
            </label>
          {/if}
        </div>
      </div>
      
      <!-- Profile Info -->
      <div class="profile-info">
        {#if isEditing}
          <input 
            type="text" 
            bind:value={editData.name}
            class="edit-name"
            style="color: {colors.text}; border-color: {colors.border}"
          />
        {:else}
          <h1 class="unit-name" style="color: {colors.text}">{user?.name}</h1>
        {/if}
        
        <div class="unit-details">
          <span class="unit-id" style="color: {colors.text}">ID: {user?.unitId}</span>
          <span class="unit-type" style="color: {colors.text}">{user?.unitType?.toUpperCase()}</span>
          <span class="unit-rank" style="background-color: {colors.success}; color: {colors.bg}">
            RANK {user?.rank}
          </span>
        </div>
        
        <!-- Level Progress -->
        <div class="level-section">
          <div class="level-info">
            <span style="color: {colors.text}">LEVEL {user?.level}</span>
            <span style="color: {colors.text}">{user?.xp} / {getMaxXp(user?.level || 1)} XP</span>
          </div>
          <div class="xp-bar" style="border-color: {colors.border}; background-color: {colors.bg}">
            <div class="xp-progress" style="width: {xpProgress}%; background-color: {colors.text}"></div>
          </div>
        </div>
        
        <!-- Bio -->
        {#if isEditing}
          <textarea 
            bind:value={editData.bio}
            class="edit-bio"
            style="color: {colors.text}; border-color: {colors.border}"
            rows="3"
          />
        {:else}
          <p class="unit-bio" style="color: {colors.text}">{user?.bio}</p>
        {/if}
      </div>
      
      <!-- Edit Controls -->
      <div class="edit-controls">
        {#if !isEditing}
          <button on:click={startEdit} class="edit-btn" style="border-color: {colors.border}">
            <span style="color: {colors.text}">✏️</span>
          </button>
        {:else}
          <button on:click={saveProfile} class="save-btn" style="background-color: {colors.success}">
            <span style="color: {colors.bg}">💾</span>
          </button>
          <button on:click={cancelEdit} class="cancel-btn" style="border-color: {colors.error}">
            <span style="color: {colors.error}">✖️</span>
          </button>
        {/if}
      </div>
    </div>
  </div>
  
  <!-- Tabs -->
  <div class="tabs" style="border-color: {colors.border}">
    {#each ['overview', 'activity', 'achievements', 'equipment', 'settings'] as tab}
      <button 
        class="tab"
        class:active={activeTab === tab}
        on:click={() => activeTab = tab}
        style="color: {activeTab === tab ? colors.bg : colors.text}; 
               background-color: {activeTab === tab ? colors.text : 'transparent'};
               border-color: {colors.border}"
      >
        {tab.toUpperCase()}
      </button>
    {/each}
  </div>
  
  <!-- Tab Content -->
  <div class="tab-content" style="background-color: {colors.highlight}; border-color: {colors.border}">
    {#if activeTab === 'overview'}
      <div class="overview-grid">
        <!-- Stats Card -->
        <div class="stats-card" style="border-color: {colors.border}; background-color: {colors.bg}">
          <h3 style="color: {colors.text}">UNIT STATISTICS</h3>
          <div class="stats-grid">
            <div class="stat">
              <span class="stat-label" style="color: {colors.text}">Missions Completed</span>
              <span class="stat-value" style="color: {colors.text}">{user?.missionsCompleted}</span>
            </div>
            <div class="stat">
              <span class="stat-label" style="color: {colors.text}">Combat Rating</span>
              <span class="stat-value" style="color: {colors.text}">{user?.combatRating}%</span>
            </div>
            <div class="stat">
              <span class="stat-label" style="color: {colors.text}">Hours Active</span>
              <span class="stat-value" style="color: {colors.text}">{user?.hoursActive}</span>
            </div>
            <div class="stat">
              <span class="stat-label" style="color: {colors.text}">Achievements</span>
              <span class="stat-value" style="color: {colors.text}">{user?.achievementsUnlocked}/60</span>
            </div>
          </div>
        </div>
        
        <!-- Info Card -->
        <div class="info-card" style="border-color: {colors.border}; background-color: {colors.bg}">
          <h3 style="color: {colors.text}">UNIT INFORMATION</h3>
          <div class="info-list">
            <div class="info-row">
              <span style="color: {colors.text}">📧 Email:</span>
              <span style="color: {colors.text}">{user?.email}</span>
            </div>
            <div class="info-row">
              <span style="color: {colors.text}">📅 Deployment Date:</span>
              <span style="color: {colors.text}">{formatDate(user?.createdAt || '')}</span>
            </div>
            <div class="info-row">
              <span style="color: {colors.text}">🕒 Last Active:</span>
              <span style="color: {colors.text}">{timeAgo(user?.lastLoginAt || user?.createdAt || '')}</span>
            </div>
            <div class="info-row">
              <span style="color: {colors.text}">✅ Email Verified:</span>
              <span style="color: {user?.emailVerified ? colors.success : colors.error}">
                {user?.emailVerified ? 'YES' : 'NO'}
              </span>
            </div>
            <div class="info-row">
              <span style="color: {colors.text}">🔐 2FA Enabled:</span>
              <span style="color: {user?.twoFactorEnabled ? colors.success : colors.error}">
                {user?.twoFactorEnabled ? 'YES' : 'NO'}
              </span>
            </div>
          </div>
        </div>
      </div>
      
    {:else if activeTab === 'activity'}
      <div class="activity-list">
        <h3 style="color: {colors.text}">RECENT ACTIVITY</h3>
        {#if activities && activities.length > 0}
          <div class="activity-items">
            {#each activities as activity}
              <div class="activity-item" style="border-color: {colors.border}">
                <span class="activity-icon">{activityIcons[activity.activityType] || '📝'}</span>
                <div class="activity-content">
                  <span class="activity-desc" style="color: {colors.text}">{activity.description}</span>
                  <span class="activity-time" style="color: {colors.text}">{timeAgo(activity.createdAt)}</span>
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <p style="color: {colors.text}">No recent activity</p>
        {/if}
      </div>
      
    {:else if activeTab === 'achievements'}
      <div class="achievements-grid">
        <h3 style="color: {colors.text}">ACHIEVEMENTS</h3>
        {#if achievements && achievements.length > 0}
          <div class="achievement-items">
            {#each achievements as item}
              <div 
                class="achievement-card"
                class:unlocked={item.unlockedAt}
                style="border-color: {colors.border}; background-color: {colors.bg}"
              >
                <span class="achievement-icon">{item.achievement?.icon || '🏆'}</span>
                <div class="achievement-info">
                  <span class="achievement-name" style="color: {colors.text}">
                    {item.achievement?.name}
                  </span>
                  {#if item.unlockedAt}
                    <span class="achievement-date" style="color: {colors.success}">
                      Unlocked {formatDate(item.unlockedAt)}
                    </span>
                  {:else}
                    <div class="achievement-progress">
                      <span style="color: {colors.text}">Progress: {item.progress}%</span>
                      <div class="progress-bar" style="border-color: {colors.border}">
                        <div class="progress-fill" style="width: {item.progress}%; background-color: {colors.accent}"></div>
                      </div>
                    </div>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <p style="color: {colors.text}">No achievements yet</p>
        {/if}
      </div>
      
    {:else if activeTab === 'equipment'}
      <div class="equipment-list">
        <h3 style="color: {colors.text}">EQUIPMENT</h3>
        {#if equipment && equipment.length > 0}
          <div class="equipment-items">
            {#each equipment as item}
              <div class="equipment-card" style="border-color: {colors.border}; background-color: {colors.bg}">
                <div class="equipment-header">
                  <span class="equipment-name" style="color: {colors.text}">
                    {item.equipment?.name}
                  </span>
                  {#if item.equipped}
                    <span class="equipped-badge" style="background-color: {colors.success}; color: {colors.bg}">
                      EQUIPPED
                    </span>
                  {/if}
                </div>
                <div class="equipment-details">
                  <span style="color: {colors.text}">Level: {item.level}</span>
                  <span style="color: {colors.text}">Type: {item.equipment?.type}</span>
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <p style="color: {colors.text}">No equipment acquired</p>
        {/if}
      </div>
      
    {:else if activeTab === 'settings'}
      <div class="settings-section">
        <h3 style="color: {colors.text}">SETTINGS</h3>
        <div class="settings-groups">
          <div class="settings-group">
            <h4 style="color: {colors.text}">Privacy Settings</h4>
            <div class="setting-item">
              <label style="color: {colors.text}">
                <input type="checkbox" checked={user?.settings?.notifications} />
                Enable Notifications
              </label>
            </div>
            <div class="setting-item">
              <label style="color: {colors.text}">
                <input type="checkbox" checked={user?.settings?.showActivityStatus} />
                Show Activity Status
              </label>
            </div>
            <div class="setting-item">
              <label style="color: {colors.text}">
                <input type="checkbox" checked={user?.settings?.dataCollection} />
                Allow Data Collection
              </label>
            </div>
          </div>
          
          <div class="settings-group">
            <h4 style="color: {colors.text}">Security Settings</h4>
            <button class="settings-btn" style="border-color: {colors.border}; color: {colors.text}">
              {user?.twoFactorEnabled ? 'Disable' : 'Enable'} Two-Factor Authentication
            </button>
            <button class="settings-btn" style="border-color: {colors.border}; color: {colors.text}">
              Change Password
            </button>
            <button class="settings-btn danger" style="border-color: {colors.error}; color: {colors.error}">
              Delete Account
            </button>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .profile-container {
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 1rem;
  }
  
  .profile-header {
    border: 2px solid;
    padding: 2rem;
    margin-bottom: 2rem;
  }
  
  .profile-header-content {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 2rem;
    align-items: start;
  }
  
  .avatar-section {
    position: relative;
  }
  
  .avatar-container {
    width: 128px;
    height: 128px;
    border: 2px solid;
    overflow: hidden;
    position: relative;
  }
  
  .avatar-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .avatar-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .avatar-icon {
    font-size: 4rem;
  }
  
  .avatar-upload {
    position: absolute;
    bottom: 0;
    right: 0;
    padding: 0.5rem;
    cursor: pointer;
    border: 1px solid;
  }
  
  .profile-info {
    flex: 1;
  }
  
  .unit-name {
    font-size: 2rem;
    font-weight: bold;
    margin: 0 0 0.5rem 0;
    font-family: monospace;
  }
  
  .edit-name {
    font-size: 2rem;
    font-weight: bold;
    font-family: monospace;
    background: transparent;
    border: none;
    border-bottom: 2px solid;
    outline: none;
    width: 100%;
    margin-bottom: 0.5rem;
  }
  
  .unit-details {
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1rem;
    font-family: monospace;
    font-size: 0.875rem;
  }
  
  .unit-rank {
    padding: 0.25rem 0.5rem;
    font-weight: bold;
  }
  
  .level-section {
    margin: 1rem 0;
  }
  
  .level-info {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-family: monospace;
    font-size: 0.875rem;
  }
  
  .xp-bar {
    height: 8px;
    border: 1px solid;
    position: relative;
  }
  
  .xp-progress {
    height: 100%;
    transition: width 0.3s ease;
  }
  
  .unit-bio {
    margin: 1rem 0 0 0;
    line-height: 1.5;
  }
  
  .edit-bio {
    width: 100%;
    padding: 0.5rem;
    background: transparent;
    border: 1px solid;
    outline: none;
    font-family: inherit;
    resize: vertical;
  }
  
  .edit-controls {
    display: flex;
    gap: 0.5rem;
  }
  
  .edit-btn,
  .save-btn,
  .cancel-btn {
    padding: 0.5rem;
    border: 1px solid;
    background: transparent;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .edit-btn:hover,
  .save-btn:hover,
  .cancel-btn:hover {
    transform: translate(1px, 1px);
  }
  
  .tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 2rem;
    border-bottom: 2px solid;
    padding-bottom: 0.5rem;
  }
  
  .tab {
    padding: 0.5rem 1rem;
    border: 1px solid;
    background: transparent;
    cursor: pointer;
    font-family: monospace;
    font-size: 0.875rem;
    transition: all 0.2s;
  }
  
  .tab:hover {
    transform: translate(1px, 1px);
  }
  
  .tab.active {
    transform: translate(1px, 1px);
  }
  
  .tab-content {
    border: 2px solid;
    padding: 2rem;
    min-height: 400px;
  }
  
  .overview-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }
  
  .stats-card,
  .info-card {
    border: 1px solid;
    padding: 1.5rem;
  }
  
  .stats-card h3,
  .info-card h3 {
    margin: 0 0 1rem 0;
    font-family: monospace;
    font-size: 1rem;
  }
  
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  
  .stat {
    display: flex;
    flex-direction: column;
  }
  
  .stat-label {
    font-size: 0.75rem;
    opacity: 0.7;
  }
  
  .stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    font-family: monospace;
  }
  
  .info-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .info-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
  }
  
  .activity-list h3 {
    margin: 0 0 1rem 0;
    font-family: monospace;
  }
  
  .activity-items {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .activity-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem;
    border-bottom: 1px solid;
  }
  
  .activity-icon {
    font-size: 1.5rem;
  }
  
  .activity-content {
    flex: 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .activity-desc {
    font-size: 0.875rem;
  }
  
  .activity-time {
    font-size: 0.75rem;
    opacity: 0.7;
    font-family: monospace;
  }
  
  .achievements-grid h3,
  .equipment-list h3,
  .settings-section h3 {
    margin: 0 0 1rem 0;
    font-family: monospace;
  }
  
  .achievement-items {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1rem;
  }
  
  .achievement-card {
    border: 1px solid;
    padding: 1rem;
    display: flex;
    gap: 1rem;
    opacity: 0.5;
  }
  
  .achievement-card.unlocked {
    opacity: 1;
  }
  
  .achievement-icon {
    font-size: 2rem;
  }
  
  .achievement-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  
  .achievement-name {
    font-weight: bold;
    font-size: 0.875rem;
  }
  
  .achievement-date {
    font-size: 0.75rem;
  }
  
  .progress-bar {
    height: 4px;
    border: 1px solid;
    margin-top: 0.25rem;
  }
  
  .progress-fill {
    height: 100%;
  }
  
  .equipment-items {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
  }
  
  .equipment-card {
    border: 1px solid;
    padding: 1rem;
  }
  
  .equipment-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }
  
  .equipment-name {
    font-weight: bold;
  }
  
  .equipped-badge {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    font-family: monospace;
  }
  
  .equipment-details {
    display: flex;
    gap: 1rem;
    font-size: 0.875rem;
    opacity: 0.8;
  }
  
  .settings-groups {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }
  
  .settings-group h4 {
    margin: 0 0 1rem 0;
    font-family: monospace;
  }
  
  .setting-item {
    margin-bottom: 0.75rem;
  }
  
  .setting-item label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    font-size: 0.875rem;
  }
  
  .settings-btn {
    display: block;
    width: 100%;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    border: 1px solid;
    background: transparent;
    cursor: pointer;
    font-family: monospace;
    font-size: 0.875rem;
    transition: all 0.2s;
  }
  
  .settings-btn:hover {
    transform: translate(1px, 1px);
  }
  
  @media (max-width: 768px) {
    .profile-header-content {
      grid-template-columns: 1fr;
    }
    
    .overview-grid,
    .settings-groups {
      grid-template-columns: 1fr;
    }
  }
</style>