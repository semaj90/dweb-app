<!-- Document Update Notifications Component -->
<!-- Shows real-time updates for document re-embedding and re-ranking -->

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { slide } from 'svelte/transition';
  import { 
    documentUpdateNotifications, 
    notificationManager,
    formatNotificationTime,
    getNotificationIcon,
    getPriorityColor,
    type UpdateNotification 
  } from '$lib/services/documentUpdateNotifications';

  // Props
  let { 
    showAll = false,
    maxVisible = 5,
    autoHide = true,
    position = 'top-right' 
  } = $props();

  // State
  let notifications = $state<UpdateNotification[]>([]);
  let activeUpdates = $state(new Map<string, UpdateNotification>());
  let connectionStatus = $state('disconnected');
  let showNotifications = $state(true);
  let notificationPermissionGranted = $state(false);

  // Subscribe to notifications store
  let unsubscribe: (() => void) | null = null;

  onMount(async () => {
    // Subscribe to notification updates
    unsubscribe = documentUpdateNotifications.subscribe(state => {
      notifications = state.notifications;
      activeUpdates = state.activeUpdates;
      connectionStatus = state.connectionStatus;
    });

    // Check notification permission
    if (notificationManager) {
      notificationPermissionGranted = await notificationManager.requestNotificationPermission();
    }
  });

  onDestroy(() => {
    if (unsubscribe) {
      unsubscribe();
    }
  });

  // Computed
  const visibleNotifications = $derived(() => {
    const recent = showAll ? notifications : notifications.slice(-maxVisible);
    return recent.reverse(); // Show newest first
  });

  const activeUpdatesList = $derived(() => {
    return Array.from(activeUpdates.values());
  });

  const connectionStatusIcon = $derived(() => {
    switch (connectionStatus) {
      case 'connected': return '🟢';
      case 'connecting': return '🟡';
      case 'disconnected': return '⚪';
      case 'error': return '🔴';
      default: return '⚪';
    }
  });

  // Methods
  function clearAllNotifications() {
    if (notificationManager) {
      notificationManager.clearNotifications();
    }
  }

  function toggleNotifications() {
    showNotifications = !showNotifications;
  }

  function getProgressWidth(notification: UpdateNotification): string {
    if (notification.data.progress !== undefined) {
      return `${notification.data.progress}%`;
    }
    
    if (notification.data.chunksProcessed && notification.data.totalChunks) {
      const progress = (notification.data.chunksProcessed / notification.data.totalChunks) * 100;
      return `${Math.round(progress)}%`;
    }
    
    return '0%';
  }
</script>

<!-- Notification Container -->
<div class="document-notifications fixed {position === 'top-right' ? 'top-4 right-4' : 'bottom-4 right-4'} z-50 w-96 max-w-sm">
  
  <!-- Connection Status & Toggle -->
  <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 mb-2 p-3">
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-2">
        <span class="text-lg">{connectionStatusIcon}</span>
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">
          Document Updates
        </span>
        {#if activeUpdatesList.length > 0}
          <span class="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
            {activeUpdatesList.length} active
          </span>
        {/if}
      </div>
      
      <div class="flex items-center space-x-1">
        {#if notifications.length > 0}
          <button
            onclick={clearAllNotifications}
            class="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded"
            title="Clear all"
          >
            Clear
          </button>
        {/if}
        
        <button
          onclick={toggleNotifications}
          class="text-gray-500 hover:text-gray-700 p-1 rounded"
          title={showNotifications ? 'Hide notifications' : 'Show notifications'}
        >
          {showNotifications ? '🔽' : '🔼'}
        </button>
      </div>
    </div>
    
    <!-- Connection Status Details -->
    <div class="mt-1 text-xs text-gray-500">
      Status: {connectionStatus}
      {#if !notificationPermissionGranted}
        <span class="text-orange-600">• Browser notifications disabled</span>
      {/if}
    </div>
  </div>

  <!-- Active Updates (Always Visible) -->
  {#if activeUpdatesList.length > 0}
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 mb-2">
      <div class="p-3 border-b border-gray-200 dark:border-gray-700">
        <h4 class="text-sm font-medium text-gray-700 dark:text-gray-300">
          🔄 Processing Updates
        </h4>
      </div>
      
      {#each activeUpdatesList as update (update.id)}
        <div class="p-3 border-b border-gray-100 dark:border-gray-600 last:border-b-0">
          <div class="flex items-start justify-between">
            <div class="flex-1">
              <div class="text-sm text-gray-700 dark:text-gray-300 mb-1">
                {update.data.title || `Document ${update.documentId.substring(0, 8)}...`}
              </div>
              
              {#if update.data.chunksProcessed && update.data.totalChunks}
                <div class="text-xs text-gray-500 mb-2">
                  Processing chunk {update.data.chunksProcessed} of {update.data.totalChunks}
                </div>
                
                <!-- Progress Bar -->
                <div class="w-full bg-gray-200 rounded-full h-2 mb-1">
                  <div 
                    class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style="width: {getProgressWidth(update)}"
                  ></div>
                </div>
              {:else}
                <div class="flex items-center space-x-2 text-xs text-gray-500">
                  <div class="animate-spin w-3 h-3 border border-blue-600 border-t-transparent rounded-full"></div>
                  <span>Processing...</span>
                </div>
              {/if}
            </div>
            
            <div class="text-xs text-gray-400 ml-2">
              {formatNotificationTime(update.timestamp)}
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Notification History -->
  {#if showNotifications && visibleNotifications.length > 0}
    <div 
      class="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-h-96 overflow-y-auto"
      transition:slide={{ duration: 200 }}
    >
      <div class="p-3 border-b border-gray-200 dark:border-gray-700">
        <h4 class="text-sm font-medium text-gray-700 dark:text-gray-300">
          📋 Recent Updates
        </h4>
      </div>
      
      {#each visibleNotifications as notification (notification.id)}
        <div 
          class="p-3 border-b border-gray-100 dark:border-gray-600 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
          transition:slide={{ duration: 150 }}
        >
          <div class="flex items-start justify-between">
            <div class="flex items-start space-x-2 flex-1">
              <span class="text-lg mt-0.5">
                {getNotificationIcon(notification.type)}
              </span>
              
              <div class="flex-1 min-w-0">
                <div class="text-sm text-gray-700 dark:text-gray-300 mb-1">
                  {#if notification.type === 'document_changed'}
                    Document "{notification.data.title || 'Untitled'}" was modified
                  {:else if notification.type === 'reembedding_started'}
                    Re-embedding "{notification.data.title || 'document'}"
                  {:else if notification.type === 'reembedding_complete'}
                    Completed re-embedding with {notification.data.chunksProcessed || 0} chunks
                  {:else if notification.type === 'reranking_complete'}
                    Updated {notification.data.queriesReranked || 0} search queries
                    {#if notification.data.similarityImprovement}
                      <span class="text-green-600">
                        (+{(notification.data.similarityImprovement * 100).toFixed(1)}% accuracy)
                      </span>
                    {/if}
                  {:else if notification.type === 'error'}
                    <span class="text-red-600">
                      Error: {notification.data.error}
                    </span>
                  {/if}
                </div>
                
                {#if notification.data.priority}
                  <span class="inline-block px-2 py-1 text-xs rounded-full {getPriorityColor(notification.data.priority)}">
                    {notification.data.priority} priority
                  </span>
                {/if}
              </div>
            </div>
            
            <div class="text-xs text-gray-400 ml-2 whitespace-nowrap">
              {formatNotificationTime(notification.timestamp)}
            </div>
          </div>
        </div>
      {/each}
      
      {#if !showAll && notifications.length > maxVisible}
        <div class="p-3 text-center">
          <button
            onclick={() => showAll = true}
            class="text-xs text-blue-600 hover:text-blue-800"
          >
            Show all {notifications.length} notifications
          </button>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Empty State -->
  {#if showNotifications && notifications.length === 0 && activeUpdatesList.length === 0}
    <div 
      class="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-6 text-center"
      transition:slide={{ duration: 200 }}
    >
      <div class="text-4xl mb-2">📭</div>
      <div class="text-sm text-gray-500">
        No document updates yet
      </div>
    </div>
  {/if}
</div>

<style>
  .document-notifications {
    /* Ensure notifications appear above other elements */
    z-index: 9999;
  }
  
  /* Custom scrollbar for notification history */
  .document-notifications :global(.overflow-y-auto) {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e0 transparent;
  }
  
  .document-notifications :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 4px;
  }
  
  .document-notifications :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: transparent;
  }
  
  .document-notifications :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background-color: #cbd5e0;
    border-radius: 2px;
  }
  
  .document-notifications :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background-color: #a0aec0;
  }
</style>