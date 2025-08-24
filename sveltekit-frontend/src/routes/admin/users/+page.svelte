<script lang="ts">
  import { onMount } from 'svelte';
  import { currentUser } from '$lib/auth/auth-store';
  import { AccessControl, ROLES, ROLE_HIERARCHY, type UserRole } from '$lib/auth/roles';
  import type { User } from '$lib/server/db/schema-postgres';
  
  // User management state
  let users: (User & { profile?: any })[] = [];
  let filteredUsers: typeof users = [];
  let selectedUsers: Set<string> = new Set();
  let isLoading = true;
  let showCreateModal = false;
  let showEditModal = false;
  let currentEditUser: typeof users[0] | null = null;
  
  // Filters and search
  let searchQuery = '';
  let roleFilter: UserRole | 'all' = 'all';
  let statusFilter: 'all' | 'active' | 'inactive' = 'all';
  
  // New user form
  let newUser = {
    email: '',
    firstName: '',
    lastName: '',
    role: 'viewer' as UserRole,
    password: '',
    confirmPassword: ''
  };
  
  // Pagination
  let currentPage = 1;
  let usersPerPage = 20;
  let totalPages = 1;
  
  // YoRHa styling
  const yorhaClasses = {
    card: 'bg-[#1a1a1a] border border-[#333333] p-4',
    cardHeader: 'text-[#00ff88] text-sm font-bold mb-4 tracking-wider flex items-center justify-between',
    button: 'px-4 py-2 border border-[#333333] bg-[#111111] hover:bg-[#2a2a2a] transition-colors text-sm',
    buttonPrimary: 'px-4 py-2 border border-[#00ff88] bg-[#002211] text-[#00ff88] hover:bg-[#003322] transition-colors text-sm',
    buttonDanger: 'px-4 py-2 border border-red-500 bg-red-900 text-red-100 hover:bg-red-800 transition-colors text-sm',
    input: 'bg-[#111111] border border-[#333333] px-3 py-2 text-sm w-full focus:border-[#00ff88] focus:outline-none',
    select: 'bg-[#111111] border border-[#333333] px-3 py-2 text-sm focus:border-[#00ff88] focus:outline-none',
    table: 'w-full border-collapse',
    tableHeader: 'border-b border-[#333333] text-left p-3 text-xs opacity-60 font-bold tracking-wider',
    tableCell: 'border-b border-[#222222] p-3 text-sm',
    modal: 'fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50',
    modalContent: 'bg-[#1a1a1a] border border-[#333333] p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto'
  };
  
  onMount(async () => {
    await loadUsers();
  });
  
  // Apply filters and search
  $: {
    filteredUsers = users.filter(user => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesSearch = 
          user.email.toLowerCase().includes(query) ||
          user.firstName?.toLowerCase().includes(query) ||
          user.lastName?.toLowerCase().includes(query);
        
        if (!matchesSearch) return false;
      }
      
      // Role filter
      if (roleFilter !== 'all' && user.role !== roleFilter) {
        return false;
      }
      
      // Status filter
      if (statusFilter !== 'all') {
        if (statusFilter === 'active' && !user.isActive) return false;
        if (statusFilter === 'inactive' && user.isActive) return false;
      }
      
      return true;
    });
    
    // Update pagination
    totalPages = Math.ceil(filteredUsers.length / usersPerPage);
    currentPage = Math.min(currentPage, totalPages || 1);
  }
  
  // Paginated users
  $: paginatedUsers = filteredUsers.slice(
    (currentPage - 1) * usersPerPage, 
    currentPage * usersPerPage
  );
  
  async function loadUsers() {
    try {
      isLoading = true;
      
      const response = await fetch('/api/admin/users', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        users = data.users || [];
      } else {
        console.error('Failed to load users:', await response.text());
      }
    } catch (error) {
      console.error('Error loading users:', error);
    } finally {
      isLoading = false;
    }
  }
  
  async function createUser() {
    if (newUser.password !== newUser.confirmPassword) {
      alert('Passwords do not match');
      return;
    }
    
    try {
      const response = await fetch('/api/admin/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email: newUser.email,
          firstName: newUser.firstName,
          lastName: newUser.lastName,
          role: newUser.role,
          password: newUser.password
        }),
        credentials: 'include'
      });
      
      if (response.ok) {
        await loadUsers();
        showCreateModal = false;
        resetNewUserForm();
      } else {
        const error = await response.json();
        alert(error.message || 'Failed to create user');
      }
    } catch (error) {
      console.error('Error creating user:', error);
      alert('Network error while creating user');
    }
  }
  
  async function updateUser(userId: string, updates: Partial<User>) {
    try {
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updates),
        credentials: 'include'
      });
      
      if (response.ok) {
        await loadUsers();
        showEditModal = false;
        currentEditUser = null;
      } else {
        const error = await response.json();
        alert(error.message || 'Failed to update user');
      }
    } catch (error) {
      console.error('Error updating user:', error);
      alert('Network error while updating user');
    }
  }
  
  async function toggleUserStatus(userId: string, isActive: boolean) {
    await updateUser(userId, { isActive });
  }
  
  async function deleteUser(userId: string) {
    if (!confirm('Are you sure you want to delete this user? This action cannot be undone.')) {
      return;
    }
    
    try {
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: 'DELETE',
        credentials: 'include'
      });
      
      if (response.ok) {
        await loadUsers();
      } else {
        const error = await response.json();
        alert(error.message || 'Failed to delete user');
      }
    } catch (error) {
      console.error('Error deleting user:', error);
      alert('Network error while deleting user');
    }
  }
  
  async function bulkAction(action: string) {
    if (selectedUsers.size === 0) {
      alert('No users selected');
      return;
    }
    
    try {
      const response = await fetch('/api/admin/users/bulk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          action,
          userIds: Array.from(selectedUsers)
        }),
        credentials: 'include'
      });
      
      if (response.ok) {
        await loadUsers();
        selectedUsers.clear();
        selectedUsers = selectedUsers; // Trigger reactivity
      } else {
        const error = await response.json();
        alert(error.message || 'Bulk action failed');
      }
    } catch (error) {
      console.error('Error performing bulk action:', error);
      alert('Network error during bulk action');
    }
  }
  
  function resetNewUserForm() {
    newUser = {
      email: '',
      firstName: '',
      lastName: '',
      role: 'viewer',
      password: '',
      confirmPassword: ''
    };
  }
  
  function openEditModal(user: typeof users[0]) {
    currentEditUser = { ...user };
    showEditModal = true;
  }
  
  function canManageUser(targetUser: typeof users[0]): boolean {
    if (!$currentUser) return false;
    
    // Can't manage yourself through this interface
    if (targetUser.id === $currentUser.id) return false;
    
    // Check role hierarchy
    return AccessControl.hasHigherAuthority($currentUser.role, targetUser.role);
  }
  
  function canAssignRole(role: UserRole): boolean {
    if (!$currentUser) return false;
    return AccessControl.canAssignRole($currentUser.role, role);
  }
  
  function getRoleDisplayName(role: string): string {
    return ROLES[role as UserRole]?.displayName || role.replace('_', ' ').toUpperCase();
  }
  
  function getRoleBadgeColor(role: string): string {
    const roleLevel = ROLES[role as UserRole]?.hierarchyLevel || 0;
    if (roleLevel >= 80) return 'border-red-500 text-red-400';
    if (roleLevel >= 60) return 'border-[#00ff88] text-[#00ff88]';
    if (roleLevel >= 40) return 'border-yellow-500 text-yellow-400';
    return 'border-gray-500 text-gray-400';
  }
</script>

<svelte:head>
  <title>User Management - Admin - Legal AI Platform</title>
</svelte:head>

<!-- User Management Interface -->
<div class="space-y-6">
  <!-- Header -->
  <div class={yorhaClasses.cardHeader}>
    <div>
      <h1 class="text-xl font-bold">USER MANAGEMENT</h1>
      <p class="text-xs opacity-60 mt-1">MANAGE SYSTEM USERS, ROLES, AND PERMISSIONS</p>
    </div>
    
    <div class="flex space-x-2">
      <button 
        on:click={() => showCreateModal = true}
        class={yorhaClasses.buttonPrimary}
      >
        ◈ CREATE USER
      </button>
      <button 
        on:click={loadUsers}
        class={yorhaClasses.button}
      >
        ↻ REFRESH
      </button>
    </div>
  </div>
  
  <!-- Filters and Search -->
  <div class={yorhaClasses.card}>
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <!-- Search -->
      <div>
        <label class="block text-xs opacity-60 mb-2">SEARCH USERS</label>
        <input 
          type="text" 
          bind:value={searchQuery}
          placeholder="Email, name..."
          class={yorhaClasses.input}
        >
      </div>
      
      <!-- Role Filter -->
      <div>
        <label class="block text-xs opacity-60 mb-2">FILTER BY ROLE</label>
        <select bind:value={roleFilter} class={yorhaClasses.select}>
          <option value="all">ALL ROLES</option>
          {#each ROLE_HIERARCHY as role}
            <option value={role}>{getRoleDisplayName(role)}</option>
          {/each}
        </select>
      </div>
      
      <!-- Status Filter -->
      <div>
        <label class="block text-xs opacity-60 mb-2">FILTER BY STATUS</label>
        <select bind:value={statusFilter} class={yorhaClasses.select}>
          <option value="all">ALL STATUS</option>
          <option value="active">ACTIVE</option>
          <option value="inactive">INACTIVE</option>
        </select>
      </div>
      
      <!-- Bulk Actions -->
      <div>
        <label class="block text-xs opacity-60 mb-2">BULK ACTIONS</label>
        <div class="flex space-x-1">
          <button 
            on:click={() => bulkAction('activate')}
            class="{yorhaClasses.button} flex-1"
            disabled={selectedUsers.size === 0}
          >
            ACTIVATE
          </button>
          <button 
            on:click={() => bulkAction('deactivate')}
            class="{yorhaClasses.button} flex-1"
            disabled={selectedUsers.size === 0}
          >
            DEACTIVATE
          </button>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Users Table -->
  <div class={yorhaClasses.card}>
    {#if isLoading}
      <div class="text-center py-8">
        <div class="text-4xl mb-4">◈</div>
        <div>LOADING USERS...</div>
      </div>
    {:else if paginatedUsers.length > 0}
      <div class="overflow-x-auto">
        <table class={yorhaClasses.table}>
          <thead>
            <tr>
              <th class={yorhaClasses.tableHeader}>
                <input 
                  type="checkbox" 
                  on:change={(e) => {
                    if (e.currentTarget?.checked) {
                      selectedUsers = new Set(paginatedUsers.map(u => u.id));
                    } else {
                      selectedUsers.clear();
                      selectedUsers = selectedUsers;
                    }
                  }}
                >
              </th>
              <th class={yorhaClasses.tableHeader}>STATUS</th>
              <th class={yorhaClasses.tableHeader}>USER</th>
              <th class={yorhaClasses.tableHeader}>ROLE</th>
              <th class={yorhaClasses.tableHeader}>CREATED</th>
              <th class={yorhaClasses.tableHeader}>LAST ACTIVE</th>
              <th class={yorhaClasses.tableHeader}>ACTIONS</th>
            </tr>
          </thead>
          <tbody>
            {#each paginatedUsers as user}
              <tr class="hover:bg-[#222222]">
                <!-- Checkbox -->
                <td class={yorhaClasses.tableCell}>
                  <input 
                    type="checkbox" 
                    checked={selectedUsers.has(user.id)}
                    on:change={(e) => {
                      if (e.currentTarget?.checked) {
                        selectedUsers.add(user.id);
                      } else {
                        selectedUsers.delete(user.id);
                      }
                      selectedUsers = selectedUsers;
                    }}
                  >
                </td>
                
                <!-- Status -->
                <td class={yorhaClasses.tableCell}>
                  <span class={user.isActive ? 'text-[#00ff88]' : 'text-red-500'}>
                    {user.isActive ? '◈ ACTIVE' : '◯ INACTIVE'}
                  </span>
                </td>
                
                <!-- User Info -->
                <td class={yorhaClasses.tableCell}>
                  <div>
                    <div class="font-bold">{user.email}</div>
                    {#if user.firstName || user.lastName}
                      <div class="text-xs opacity-60">
                        {user.firstName} {user.lastName}
                      </div>
                    {/if}
                  </div>
                </td>
                
                <!-- Role -->
                <td class={yorhaClasses.tableCell}>
                  <span class="px-2 py-1 border {getRoleBadgeColor(user.role)} text-xs">
                    {getRoleDisplayName(user.role)}
                  </span>
                </td>
                
                <!-- Created -->
                <td class={yorhaClasses.tableCell}>
                  {new Date(user.createdAt).toLocaleDateString()}
                </td>
                
                <!-- Last Active -->
                <td class={yorhaClasses.tableCell}>
                  <span class="opacity-60">
                    {user.updatedAt ? new Date(user.updatedAt).toLocaleDateString() : 'Never'}
                  </span>
                </td>
                
                <!-- Actions -->
                <td class={yorhaClasses.tableCell}>
                  <div class="flex space-x-1">
                    {#if canManageUser(user)}
                      <button 
                        on:click={() => openEditModal(user)}
                        class="px-2 py-1 border border-[#333333] hover:bg-[#2a2a2a] text-xs"
                        title="Edit User"
                      >
                        ✎
                      </button>
                      
                      <button 
                        on:click={() => toggleUserStatus(user.id, !user.isActive)}
                        class="px-2 py-1 border border-{user.isActive ? 'red-500' : '[#00ff88]'} text-{user.isActive ? 'red-500' : '[#00ff88]'} hover:bg-opacity-20 text-xs"
                        title={user.isActive ? 'Deactivate' : 'Activate'}
                      >
                        {user.isActive ? '◯' : '◈'}
                      </button>
                      
                      <button 
                        on:click={() => deleteUser(user.id)}
                        class="px-2 py-1 border border-red-500 text-red-500 hover:bg-red-500 hover:text-black text-xs"
                        title="Delete User"
                      >
                        ✕
                      </button>
                    {:else}
                      <span class="text-xs opacity-40">No Access</span>
                    {/if}
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      
      <!-- Pagination -->
      {#if totalPages > 1}
        <div class="flex justify-between items-center mt-4 pt-4 border-t border-[#333333]">
          <div class="text-sm opacity-60">
            Showing {(currentPage - 1) * usersPerPage + 1} to {Math.min(currentPage * usersPerPage, filteredUsers.length)} of {filteredUsers.length} users
          </div>
          
          <div class="flex space-x-2">
            <button 
              on:click={() => currentPage = Math.max(1, currentPage - 1)}
              disabled={currentPage === 1}
              class={yorhaClasses.button}
            >
              ◀ PREV
            </button>
            
            <div class="flex items-center space-x-2">
              <span class="text-sm">PAGE {currentPage} OF {totalPages}</span>
            </div>
            
            <button 
              on:click={() => currentPage = Math.min(totalPages, currentPage + 1)}
              disabled={currentPage === totalPages}
              class={yorhaClasses.button}
            >
              NEXT ▶
            </button>
          </div>
        </div>
      {/if}
    {:else}
      <div class="text-center py-8 opacity-60">
        <div class="text-4xl mb-4">◯</div>
        <div>NO USERS FOUND</div>
        <div class="text-sm mt-2">Try adjusting your search or filters</div>
      </div>
    {/if}
  </div>
</div>

<!-- Create User Modal -->
{#if showCreateModal}
  <div class={yorhaClasses.modal}>
    <div class={yorhaClasses.modalContent}>
      <div class="flex justify-between items-center mb-6">
        <h2 class="text-xl font-bold text-[#00ff88]">CREATE NEW USER</h2>
        <button 
          on:click={() => showCreateModal = false}
          class="text-2xl hover:text-red-500"
        >
          ✕
        </button>
      </div>
      
      <form on:submit|preventDefault={createUser} class="space-y-4">
        <!-- Email -->
        <div>
          <label class="block text-sm font-bold mb-2">EMAIL ADDRESS</label>
          <input 
            type="email" 
            bind:value={newUser.email}
            required
            class={yorhaClasses.input}
            placeholder="user@example.com"
          >
        </div>
        
        <!-- Names -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-bold mb-2">FIRST NAME</label>
            <input 
              type="text" 
              bind:value={newUser.firstName}
              class={yorhaClasses.input}
            >
          </div>
          
          <div>
            <label class="block text-sm font-bold mb-2">LAST NAME</label>
            <input 
              type="text" 
              bind:value={newUser.lastName}
              class={yorhaClasses.input}
            >
          </div>
        </div>
        
        <!-- Role -->
        <div>
          <label class="block text-sm font-bold mb-2">USER ROLE</label>
          <select bind:value={newUser.role} class={yorhaClasses.select}>
            {#each ROLE_HIERARCHY as role}
              {#if canAssignRole(role)}
                <option value={role}>{getRoleDisplayName(role)}</option>
              {/if}
            {/each}
          </select>
          <div class="text-xs opacity-60 mt-1">
            {ROLES[newUser.role]?.description || ''}
          </div>
        </div>
        
        <!-- Password -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-bold mb-2">PASSWORD</label>
            <input 
              type="password" 
              bind:value={newUser.password}
              required
              minlength="8"
              class={yorhaClasses.input}
            >
          </div>
          
          <div>
            <label class="block text-sm font-bold mb-2">CONFIRM PASSWORD</label>
            <input 
              type="password" 
              bind:value={newUser.confirmPassword}
              required
              class={yorhaClasses.input}
            >
          </div>
        </div>
        
        <!-- Actions -->
        <div class="flex justify-end space-x-4 pt-4">
          <button 
            type="button"
            on:click={() => showCreateModal = false}
            class={yorhaClasses.button}
          >
            CANCEL
          </button>
          <button 
            type="submit"
            class={yorhaClasses.buttonPrimary}
          >
            ◈ CREATE USER
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}

<!-- Edit User Modal -->
{#if showEditModal && currentEditUser}
  <div class={yorhaClasses.modal}>
    <div class={yorhaClasses.modalContent}>
      <div class="flex justify-between items-center mb-6">
        <h2 class="text-xl font-bold text-[#00ff88]">EDIT USER</h2>
        <button 
          on:click={() => showEditModal = false}
          class="text-2xl hover:text-red-500"
        >
          ✕
        </button>
      </div>
      
      <form on:submit|preventDefault={() => updateUser(currentEditUser.id, currentEditUser)} class="space-y-4">
        <!-- Email -->
        <div>
          <label class="block text-sm font-bold mb-2">EMAIL ADDRESS</label>
          <input 
            type="email" 
            bind:value={currentEditUser.email}
            required
            class={yorhaClasses.input}
          >
        </div>
        
        <!-- Names -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-bold mb-2">FIRST NAME</label>
            <input 
              type="text" 
              bind:value={currentEditUser.firstName}
              class={yorhaClasses.input}
            >
          </div>
          
          <div>
            <label class="block text-sm font-bold mb-2">LAST NAME</label>
            <input 
              type="text" 
              bind:value={currentEditUser.lastName}
              class={yorhaClasses.input}
            >
          </div>
        </div>
        
        <!-- Role -->
        <div>
          <label class="block text-sm font-bold mb-2">USER ROLE</label>
          <select bind:value={currentEditUser.role} class={yorhaClasses.select}>
            {#each ROLE_HIERARCHY as role}
              {#if canAssignRole(role)}
                <option value={role}>{getRoleDisplayName(role)}</option>
              {/if}
            {/each}
          </select>
        </div>
        
        <!-- Status -->
        <div>
          <label class="flex items-center space-x-2">
            <input 
              type="checkbox" 
              bind:checked={currentEditUser.isActive}
            >
            <span class="text-sm font-bold">ACTIVE USER</span>
          </label>
        </div>
        
        <!-- Actions -->
        <div class="flex justify-end space-x-4 pt-4">
          <button 
            type="button"
            on:click={() => showEditModal = false}
            class={yorhaClasses.button}
          >
            CANCEL
          </button>
          <button 
            type="submit"
            class={yorhaClasses.buttonPrimary}
          >
            ◈ UPDATE USER
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}