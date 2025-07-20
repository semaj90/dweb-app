import { writable, derived } from "svelte/store";
import { browser } from "$app/environment";

// Theme system
export const theme = writable<"light" | "dark" | "auto">("auto");
export const colorScheme = writable<"blue" | "green" | "purple" | "orange">(
  "blue",
);

export type NotificationData = {
  type: "success" | "error" | "warning" | "info";
  title: string;
  message: string;
  duration?: number;
};

export type Notification = NotificationData & {
  id: string;
};

// UI State stores
export const notifications = writable<Notification[]>([]);

export const modals = writable<{
  [key: string]: boolean;
}>({});

export const loading = writable<{
  [key: string]: boolean;
}>({});

export const sidebar = writable({
  isOpen: false,
  width: 280,
  collapsed: false,
});

// Animation preferences
export const motion = writable({
  reduceMotion: false,
  duration: "normal" as "fast" | "normal" | "slow",
  spring: true,
});

// Component state
export const forms = writable<{
  [formId: string]: {
    isDirty: boolean;
    isValid: boolean;
    errors: Record<string, string>;
    touched: Record<string, boolean>;
  };
}>({});

// Derived stores
export const isDarkMode = derived(theme, ($theme) => {
  if (browser) {
    if ($theme === "auto") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return $theme === "dark";
  }
  return false;
});

export type FormField = {
  value: any;
  error?: string;
  touched?: boolean;
  isRequired?: boolean;
};

export type FormState = {
  fields: Record<string, FormField>;
  isDirty: boolean;
  isValid: boolean;
  submitCount: number;
  errors: Record<string, string>;
  values: Record<string, any>;
};

// Store actions
export const uiStore = {
  // Notifications
  notify: (notification: NotificationData) => {
    const id = crypto.randomUUID();
    const fullNotification: Notification = { ...notification, id };
    notifications.update((list) => [...list, fullNotification]);

    if (notification.duration !== 0) {
      setTimeout(() => {
        notifications.update((list) => list.filter((n) => n.id !== id));
      }, notification.duration || 5000);
    }
    return id;
  },

  dismissNotification: (id: string) => {
    notifications.update((list) => list.filter((n) => n.id !== id));
  },

  // Modals
  openModal: (modalId: string) => {
    modals.update((state) => ({ ...state, [modalId]: true }));
  },

  closeModal: (modalId: string) => {
    modals.update((state) => ({ ...state, [modalId]: false }));
  },

  // Loading states
  setLoading: (key: string, isLoading: boolean) => {
    loading.update((state) => ({ ...state, [key]: isLoading }));
  },

  // Sidebar
  toggleSidebar: () => {
    sidebar.update((state) => ({ ...state, isOpen: !state.isOpen }));
  },

  // Forms
  updateForm: (formId: string, updates: Partial<FormState>) => {
    forms.update((state) => ({
      ...state,
      [formId]: { ...state[formId], ...updates },
    }));
  },
};
