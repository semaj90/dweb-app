<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { notifications, type Notification } from "$lib/stores/notification";
  import { FocusManager } from "$lib/utils/accessibility";
  import { AlertCircle, AlertTriangle, Check, Info, X } from "lucide-svelte";
  import { createEventDispatcher, onMount } from "svelte";

  const dispatch = createEventDispatcher();

  let container: HTMLElement;
  let notificationElements = new Map<string, HTMLElement>();
  let isVisible = false;
  let maxVisible = 5;
  let position:
    | "top-right"
    | "top-left"
    | "bottom-right"
    | "bottom-left"
    | "top-center"
    | "bottom-center" = "top-right";
  let stackDirection: "up" | "down" = "down" as "up" | "down";
  let pauseOnHover = true;
  let groupSimilar = true;
  let enableSounds = true;

  // Reactive notifications list
  $: visibleNotifications = $notifications.notifications.slice(0, maxVisible);
  $: hiddenCount = Math.max(
    0,
    $notifications.notifications.length - maxVisible
  );

  onMount(() => {
    // Announce notifications to screen readers
    const unsubscribe = notifications.subscribe((notifs) => {
      const latestNotification = notifs.notifications[0];
      if (latestNotification && notifs.notifications.length > 0) {
        announceNotification(latestNotification);
        playNotificationSound(latestNotification.type);
      }
    });

    return unsubscribe;
  });

  function announceNotification(notification: Notification) {
    const message = `${notification.type} notification: ${notification.title}. ${notification.message}`;
    FocusManager.announceToScreenReader(
      message,
      notification.type === "error" ? "assertive" : "polite"
    );
  }

  function playNotificationSound(type: Notification["type"]) {
    if (!enableSounds) return;

    // Create audio context for accessibility-friendly sound feedback
    try {
      const audioContext = new (window.AudioContext ||
        (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      // Different frequencies for different notification types
      const frequencies = {
        success: 800,
        error: 400,
        warning: 600,
        info: 500,
      };

      oscillator.frequency.setValueAtTime(
        frequencies[type],
        audioContext.currentTime
      );
      oscillator.type = "sine";

      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(
        0.01,
        audioContext.currentTime + 0.2
      );

      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.2);
    } catch (error) {
      // Fallback to no sound if audio context fails
      console.debug("Audio notification unavailable:", error);
    }
  }

  function dismissNotification(id: string) {
    notifications.remove(id);
    notificationElements.delete(id);
    dispatch("dismiss", { id });
  }

  function dismissAll() {
    notifications.clear();
    notificationElements.clear();
    dispatch("dismissAll");
  }

  // Action to set notification element in the Map
  function setNotificationElement(node: HTMLElement, notificationId: string) {
    notificationElements.set(notificationId, node);

    return {
      destroy() {
        notificationElements.delete(notificationId);
      },
    };
  }

  function getNotificationIcon(type: Notification["type"]) {
    switch (type) {
      case "success":
        return Check;
      case "error":
        return AlertCircle;
      case "warning":
        return AlertTriangle;
      case "info":
      default:
        return Info;
    }
  }

  function getNotificationColor(type: Notification["type"]) {
    switch (type) {
      case "success":
        return "bg-green-50 border-green-200 text-green-800";
      case "error":
        return "bg-red-50 border-red-200 text-red-800";
      case "warning":
        return "bg-yellow-50 border-yellow-200 text-yellow-800";
      case "info":
      default:
        return "bg-blue-50 border-blue-200 text-blue-800";
    }
  }

  function handleNotificationAction(notification: Notification, action: unknown) {
    if (action.callback) {
      action.callback();
    }

    if (action.dismissOnClick !== false) {
      dismissNotification(notification.id);
    }
  }

  function pauseTimer(notification: Notification) {
    // Timer functionality could be implemented here if needed
    // For now, this is a placeholder
  }

  function resumeTimer(notification: Notification) {
    // Timer functionality could be implemented here if needed
    // For now, this is a placeholder
  }

  function getContainerClasses() {
    const baseClasses = "fixed z-50 pointer-events-none";

    switch (position) {
      case "top-right":
        return `${baseClasses} top-4 right-4`;
      case "top-left":
        return `${baseClasses} top-4 left-4`;
      case "bottom-right":
        return `${baseClasses} bottom-4 right-4`;
      case "bottom-left":
        return `${baseClasses} bottom-4 left-4`;
      case "top-center":
        return `${baseClasses} top-4 left-1/2 transform -translate-x-1/2`;
      case "bottom-center":
        return `${baseClasses} bottom-4 left-1/2 transform -translate-x-1/2`;
      default:
        return `${baseClasses} top-4 right-4`;
    }
  }

  function getAnimationClasses() {
    const isTop = position.includes("top");
    const enterFrom = isTop ? "-translate-y-2" : "translate-y-2";

    return {
      enter: `transition-all duration-300 ease-out transform ${enterFrom} opacity-0`,
      enterActive: "transform translate-y-0 opacity-100",
      exit: `transition-all duration-200 ease-in transform ${enterFrom} opacity-0`,
    };
  }
</script>

<div
  class={getContainerClasses()}
  bind:this={container}
  role="region"
  aria-label="Notifications"
  aria-live="polite"
  aria-atomic="false"
>
  {#if hiddenCount > 0}
    <div class="mx-auto px-4 max-w-7xl">
      <Button
        variant="ghost"
        size="sm"
        onclick={() => (maxVisible += 5)}
        class="mx-auto px-4 max-w-7xl"
      >
        +{hiddenCount} more notifications
      </Button>
    </div>
  {/if}

  <div
    class="mx-auto px-4 max-w-7xl"
  >
    {#each visibleNotifications as notification (notification.id)}
      <div
        class="mx-auto px-4 max-w-7xl"
        use:setNotificationElement={notification.id}
        role="alert"
        aria-labelledby="notification-title-{notification.id}"
        aria-describedby="notification-message-{notification.id}"
        onmouseenter={() => pauseTimer(notification)}
        onmouseleave={() => resumeTimer(notification)}
        onfocusin={() => pauseTimer(notification)}
        onfocusout={() => resumeTimer(notification)}
      >
        <div
          class="mx-auto px-4 max-w-7xl"
        >
          <div class="mx-auto px-4 max-w-7xl">
            <!-- Icon -->
            <div class="mx-auto px-4 max-w-7xl">
              <svelte:component
                this={getNotificationIcon(notification.type)}
                class="mx-auto px-4 max-w-7xl"
                aria-hidden="true"
              />
            </div>

            <!-- Content -->
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">
                  <p
                    id="notification-title-{notification.id}"
                    class="mx-auto px-4 max-w-7xl"
                  >
                    {notification.title}
                  </p>

                  {#if notification.message}
                    <p
                      id="notification-message-{notification.id}"
                      class="mx-auto px-4 max-w-7xl"
                    >
                      {notification.message}
                    </p>
                  {/if}

                  <!-- Progress bar for timed notifications -->
                  {#if notification.duration && notification.duration > 0}
                    <div
                      class="mx-auto px-4 max-w-7xl"
                    >
                      <div
                        class="mx-auto px-4 max-w-7xl"
                        style="width: 100%"
                      ></div>
                    </div>
                  {/if}

                  <!-- Actions -->
                  {#if notification.actions && notification.actions.length > 0}
                    <div class="mx-auto px-4 max-w-7xl">
                      {#each notification.actions as action}
                        <Button
                          size="sm"
                          variant={action.variant === "primary"
                            ? "default"
                            : "ghost"}
                          onclick={() =>
                            handleNotificationAction(notification, action)}
                          class="mx-auto px-4 max-w-7xl"
                        >
                          {action.label}
                        </Button>
                      {/each}
                    </div>
                  {/if}
                </div>

                <!-- Dismiss button -->
                <div class="mx-auto px-4 max-w-7xl">
                  <Button
                    variant="ghost"
                    size="sm"
                    onclick={() => dismissNotification(notification.id)}
                    class="mx-auto px-4 max-w-7xl"
                    aria-label="Dismiss notification"
                  >
                    <X class="mx-auto px-4 max-w-7xl" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    {/each}
  </div>

  <!-- Dismiss all button for multiple notifications -->
  {#if visibleNotifications.length > 1}
    <div class="mx-auto px-4 max-w-7xl">
      <Button
        variant="ghost"
        size="sm"
        onclick={() => dismissAll()}
        class="mx-auto px-4 max-w-7xl"
      >
        Clear all ({$notifications.notifications.length})
      </Button>
    </div>
  {/if}
</div>

<!-- Notification settings (can be toggled via settings page) -->
{#if false}
  <div class="mx-auto px-4 max-w-7xl">
    <h3 class="mx-auto px-4 max-w-7xl">Notification Settings</h3>

    <div class="mx-auto px-4 max-w-7xl">
      <label for="pause-on-hover" class="mx-auto px-4 max-w-7xl">
        <input
          id="pause-on-hover"
          type="checkbox"
          bind:checked={pauseOnHover}
        />
        <span class="mx-auto px-4 max-w-7xl">Pause on hover</span>
      </label>

      <label for="enable-sounds" class="mx-auto px-4 max-w-7xl">
        <input id="enable-sounds" type="checkbox" bind:checked={enableSounds} />
        <span class="mx-auto px-4 max-w-7xl">Enable sounds</span>
      </label>

      <label for="group-similar" class="mx-auto px-4 max-w-7xl">
        <input id="group-similar" type="checkbox" bind:checked={groupSimilar} />
        <span class="mx-auto px-4 max-w-7xl">Group similar notifications</span>
      </label>

      <div>
        <label for="max-visible-range" class="mx-auto px-4 max-w-7xl"
          >Max visible</label
        >
        <input
          id="max-visible-range"
          type="range"
          min="1"
          max="10"
          bind:value={maxVisible}
          class="mx-auto px-4 max-w-7xl"
        />
        <span class="mx-auto px-4 max-w-7xl">{maxVisible} notifications</span>
      </div>

      <div>
        <label for="position-select" class="mx-auto px-4 max-w-7xl"
          >Position</label
        >
        <select
          id="position-select"
          bind:value={position}
          class="mx-auto px-4 max-w-7xl"
        >
          <option value="top-right">Top Right</option>
          <option value="top-left">Top Left</option>
          <option value="bottom-right">Bottom Right</option>
          <option value="bottom-left">Bottom Left</option>
          <option value="top-center">Top Center</option>
          <option value="bottom-center">Bottom Center</option>
        </select>
      </div>
    </div>
  </div>
{/if}

<style>
  .notification-item {
    transform-origin: center;
  }

  /* Animations for notification entrance/exit */
  @keyframes notification-enter {
    from {
      opacity: 0;
      transform: translateY(-1rem) scale(0.95);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  @keyframes notification-exit {
    from {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
    to {
      opacity: 0;
      transform: translateY(-1rem) scale(0.95);
    }
  }

  .notification-item {
    animation: notification-enter 0.3s ease-out;
  }

  /* Reduce motion for accessibility */
  @media (prefers-reduced-motion: reduce) {
    .notification-item {
      animation: none;
    }

    .transition-all {
      transition: none !important;
    }
  }

  /* High contrast mode support */
  @media (prefers-contrast: high) {
    .notification-item {
      border-width: 2px;
    }
  }

  /* Focus indicators */
  .notification-item:focus-within {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
  }

  /* Screen reader only content */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>