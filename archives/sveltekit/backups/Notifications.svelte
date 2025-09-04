<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { quintOut } from "svelte/easing";
  import { fly } from "svelte/transition";
  import { notifications, type Notification } from "../../stores/notification";

  // Icons for different notification types
  const icons = {
    success: "ph:check-circle",
    error: "ph:x-circle",
    warning: "ph:warning-circle",
    info: "ph:info",
  };

  const colorClasses = {
    success:
      "bg-green-50 border-green-200 text-green-800 dark:bg-green-900/10 dark:border-green-800 dark:text-green-200",
    error:
      "bg-red-50 border-red-200 text-red-800 dark:bg-red-900/10 dark:border-red-800 dark:text-red-200",
    warning:
      "bg-yellow-50 border-yellow-200 text-yellow-800 dark:bg-yellow-900/10 dark:border-yellow-800 dark:text-yellow-200",
    info: "bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-900/10 dark:border-blue-800 dark:text-blue-200",
  };

  const iconColorClasses = {
    success: "text-green-400",
    error: "text-red-400",
    warning: "text-yellow-400",
    info: "text-blue-400",
  };

  function handleClose(notification: Notification) {
    notifications.remove(notification.id);
}
  function handleAction(
    notification: Notification,
    action: NonNullable<Notification["actions"]>[0]
  ) {
    action.action();
    notifications.remove(notification.id);
}
</script>

<!-- Notification Container -->
<div class="mx-auto px-4 max-w-7xl">
  {#each $notifications.notifications as notification (notification.id)}
    <div
      class={`
				relative p-4 rounded-lg border shadow-lg backdrop-blur-sm
				${colorClasses[notification.type]}
			`}
      in:fly={{ x: 300, duration: 300, easing: quintOut "
      out:fly={{ x: 300, duration: 200, easing: quintOut  "
    >
      <div class="mx-auto px-4 max-w-7xl">
        <!-- Icon -->
        <div class="mx-auto px-4 max-w-7xl">
          <iconify-icon
            icon={icons[notification.type]}
            class={`w-5 h-5 ${iconColorClasses[notification.type]}`}
          ></iconify-icon>
        </div>

        <!-- Content -->
        <div class="mx-auto px-4 max-w-7xl">
          <p class="mx-auto px-4 max-w-7xl">
            {notification.title}
          </p>
          {#if notification.message}
            <p class="mx-auto px-4 max-w-7xl">
              {notification.message}
            </p>
          {/if}

          <!-- Actions -->
          {#if notification.actions && notification.actions.length > 0}
            <div class="mx-auto px-4 max-w-7xl">
              {#each notification.actions as action}
                <Button
                  size="sm"
                  variant={action.variant || "ghost"}
                  onclick={() => handleAction(notification, action)}
                >
                  {action.label}
                </Button>
              {/each}
            </div>
          {/if}
        </div>

        <!-- Close button -->
        <div class="mx-auto px-4 max-w-7xl">
          <button
            type="button"
            class="mx-auto px-4 max-w-7xl"
            onclick={() => handleClose(notification)}
          >
            <span class="mx-auto px-4 max-w-7xl">Dismiss</span>
            <iconify-icon data-icon="${1}" class="mx-auto px-4 max-w-7xl"></iconify-icon>
          </button>
        </div>
      </div>

      <!-- Progress bar for timed notifications -->
      {#if notification.duration && notification.duration > 0}
        <div
          class="mx-auto px-4 max-w-7xl"
        >
          <div
            class="mx-auto px-4 max-w-7xl"
            style="animation: shrink {notification.duration}ms linear forwards;"
          ></div>
        </div>
      {/if}
    </div>
  {/each}
</div>

<style>
  @keyframes shrink {
    from {
      width: 100%;
}
    to {
      width: 0%;
}}
</style>
