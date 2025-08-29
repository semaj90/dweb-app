import { writable } from "svelte/store";


export interface Notification {
  id: string;
  type: "success" | "error" | "warning" | "info";
  title: string;
  message?: string;
  duration?: number;
  actions?: Array<{
    label: string;
    action: () => void;
    variant?: "primary" | "secondary";
  }>;
}

export interface NotificationState {
  notifications: Notification[];
}

const initialState: NotificationState = {
  notifications: [],
};

function createNotificationStore() {
  const { subscribe, set, update } = writable<NotificationState>(initialState);

  const store = {
    subscribe,

    // Add a notification
    add: (notification: Omit<Notification, "id">) => {
      const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const newNotification: Notification = {
        ...notification,
        id,
        duration: notification.duration ?? 5000,
      };

      update((state) => ({
        notifications: [...state.notifications, newNotification],
      }));

      // Auto-remove after duration (unless duration is 0)
      if ((newNotification.duration ?? 0) > 0) {
        setTimeout(() => {
          store.remove(id);
        }, newNotification.duration);
      }
      return id;
    },

    // Remove a notification
    remove: (id: string) => {
      update((state) => ({
        notifications: state.notifications.filter((n) => n.id !== id),
      }));
    },

    // Clear all notifications
    clear: () => {
      set(initialState);
    },

    // Convenience methods
    success: (
      title: string,
      message?: string,
      options?: Partial<Notification>,
    ) => {
      return store.add({ type: "success", title, message, ...options });
    },

    error: (
      title: string,
      message?: string,
      options?: Partial<Notification>,
    ) => {
      return store.add({
        type: "error",
        title,
        message,
        duration: 0,
        ...options,
      });
    },

    warning: (
      title: string,
      message?: string,
      options?: Partial<Notification>,
    ) => {
      return store.add({ type: "warning", title, message, ...options });
    },

    info: (
      title: string,
      message?: string,
      options?: Partial<Notification>,
    ) => {
      return store.add({ type: "info", title, message, ...options });
    },

    // Legacy compatibility methods that accept objects without title
    addLegacy: (notification: {
      type: "success" | "error" | "warning" | "info";
      message: string;
      timeout?: number;
      duration?: number;
    }) => {
      const title =
        notification.type.charAt(0).toUpperCase() + notification.type.slice(1);
      return store.add({
        type: notification.type,
        title,
        message: notification.message,
        duration: notification.timeout || notification.duration,
      });
    },
  };

  return store;
}

export const notifications = createNotificationStore();
export default notifications;
