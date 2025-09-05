<!-- YoRHa Notification Manager Component -->
<script lang="ts">
  import YoRHaNotification from './YoRHaNotification.svelte';
  import { notificationStore as notificationStoreExport } from '$lib/stores/notifications';

  interface Notification {
    id: string
    type: 'info' | 'success' | 'warning' | 'error' | 'system';
    title?: string;
    message: string
    duration?: number;
    persistent?: boolean;
    closable?: boolean;
    icon?: string;
    position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'center';
    showProgress?: boolean;
  }

  // Subscribe to notification store
  let notifications = $state<Notification[]>([]);
  // Avoid TS union issues by creating a typed alias
  const notificationStore = notificationStoreExport as unknown as {
    subscribe: (run: (value: Notification[]) => void) => () => void;
    remove: (id: string) => void;
  };

  $effect(() => {
    const unsubscribe = notificationStore.subscribe((value) => {
      notifications = value;
    });

    return unsubscribe;
  });

  function removeNotification(id: string) {
    notificationStore.remove(id);
  }

  // Group notifications by position for proper stacking
  function groupNotificationsByPosition(notifications: Notification[]) {
    return notifications.reduce((groups, notification) => {
      const position = notification.position || 'top-right';
      if (!groups[position]) {
        groups[position] = [];
      }
      groups[position].push(notification);
      return groups;
    }, {} as Record<string, Notification[]>);
  }

  const groupedNotifications = $derived(groupNotificationsByPosition(notifications));
</script>

<!-- Render notifications grouped by position -->
{#each Object.entries(groupedNotifications) as [position, notificationGroup]}
  <div class="notification-group notification-group-{position}">
    {#each notificationGroup as notification (notification.id)}
      <YoRHaNotification
        {...notification}
        onclose={() => removeNotification(notification.id)}
      />
    {/each}
  </div>
{/each}

<style>
  .notification-group {
    position: fixed
    z-index: 9999;
    display: flex
    flex-direction: column
    gap: 12px;
    pointer-events: none
  }

  .notification-group > :global(.yorha-notification) {
    pointer-events: auto
  }

  /* Position-specific stacking */
  .notification-group-top-right {
    top: 20px;
    right: 20px;
  }

  .notification-group-top-left {
    top: 20px;
    left: 20px;
  }

  .notification-group-bottom-right {
    bottom: 20px;
    right: 20px;
    flex-direction: column-reverse;
  }

  .notification-group-bottom-left {
    bottom: 20px;
    left: 20px;
    flex-direction: column-reverse;
  }

  .notification-group-center {
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    align-items: center
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .notification-group-top-right,
    .notification-group-top-left {
      top: 10px;
      right: 10px;
      left: 10px;
    }

    .notification-group-bottom-right,
    .notification-group-bottom-left {
      bottom: 10px;
      right: 10px;
      left: 10px;
    }
  }
</style>