import { writable, type Writable } from 'svelte/store';

export type N64ToastType = 'info' | 'success' | 'warning' | 'error';

export interface N64Toast {
  id: string;
  message: string;
  type?: N64ToastType;
  timeout?: number; // ms, undefined or 0 means do not auto-dismiss
  createdAt: number;
}

export interface N64ToastInput {
  id?: string;
  message: string;
  type?: N64ToastType;
  timeout?: number;
}

const DEFAULT_TIMEOUT = 4000;

function generateId(): string {
  return Math.random().toString(36).slice(2, 9);
}

const toastsStore: Writable<N64Toast[]> = writable([]);

export const N64ToastStore = {
  subscribe: toastsStore.subscribe,
  /**
   * Push a new toast. Returns the generated toast id.
   */
  push(input: N64ToastInput): string {
	const id = input.id ?? generateId();
	const toast: N64Toast = {
	  id,
	  message: input.message,
	  type: input.type,
	  timeout: typeof input.timeout === 'number' ? input.timeout : DEFAULT_TIMEOUT,
	  createdAt: Date.now(),
	};

	toastsStore.update((list) => [...list, toast]);

	// Only schedule auto-remove in a browser environment to avoid SSR side-effects.
	if (typeof window !== 'undefined' && toast.timeout && toast.timeout > 0) {
	  setTimeout(() => {
		this.remove(id);
	  }, toast.timeout);
	}

	return id;
  },

  /**
   * Remove a toast by id.
   */
  remove(id: string) {
	toastsStore.update((list) => list.filter((t) => t.id !== id));
  },

  /**
   * Clear all toasts.
   */
  clear() {
	toastsStore.set([]);
  },

  /**
   * Get current toasts snapshot (not reactive).
   */
  snapshot(): N64Toast[] {
	let snap: N64Toast[] = [];
	toastsStore.subscribe((v) => (snap = v))(); // subscribe and immediately unsubscribe
	return snap;
  },
};

export default N64ToastStore;
