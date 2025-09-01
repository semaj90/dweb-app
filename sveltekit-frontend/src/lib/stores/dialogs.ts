import { writable } from 'svelte/store';

export interface Dialog {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'confirm' | 'prompt';
  title?: string;
  message?: string;
  value?: string;
  position?: 'center' | 'top' | 'bottom';
  persistent?: boolean;
  resolve?: (result: any) => void;
  reject?: (reason?: unknown) => void;
}

export interface Modal {
  id: string;
  component?: unknown;
  props?: Record<string, any>;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'fullscreen';
  type?: 'default' | 'confirm' | 'alert' | 'system';
  persistent?: boolean;
  resolve?: (result: any) => void;
  reject?: (reason?: unknown) => void;
}

function createDialogStore() {
  const { subscribe, update } = writable<Dialog[]>([]);

  function generateId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  function add(dialog: Omit<Dialog, 'id'>): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = generateId();
      const newDialog: Dialog = {
        ...dialog,
        id,
        resolve,
        reject
      };

      update(dialogs => [...dialogs, newDialog]);
    });
  }

  function remove(id: string, result?: unknown) {
    update(dialogs => {
      const dialog = dialogs.find(d => d.id === id);
      if (dialog?.resolve) {
        dialog.resolve(result);
      }
      return dialogs.filter(d => d.id !== id);
    });
  }

  function reject(id: string, reason?: unknown) {
    update(dialogs => {
      const dialog = dialogs.find(d => d.id === id);
      if (dialog?.reject) {
        dialog.reject(reason);
      }
      return dialogs.filter(d => d.id !== id);
    });
  }

  function clear() {
    update(dialogs => {
      dialogs.forEach(dialog => {
        if (dialog.reject) {
          dialog.reject('cleared');
        }
      });
      return [];
    });
  }

  // Convenience methods
  function info(title: string, message?: string): Promise<void> {
    return add({
      type: 'info',
      title,
      message,
      position: 'center'
    });
  }

  function success(title: string, message?: string): Promise<void> {
    return add({
      type: 'success',
      title,
      message,
      position: 'center'
    });
  }

  function warning(title: string, message?: string): Promise<void> {
    return add({
      type: 'warning',
      title,
      message,
      position: 'center'
    });
  }

  function error(title: string, message?: string): Promise<void> {
    return add({
      type: 'error',
      title,
      message,
      position: 'center',
      persistent: true
    });
  }

  function confirm(title: string, message?: string): Promise<boolean> {
    return add({
      type: 'confirm',
      title,
      message,
      position: 'center'
    }).then(() => true).catch(() => false);
  }

  function prompt(title: string, message?: string, defaultValue?: string): Promise<string | null> {
    return add({
      type: 'prompt',
      title,
      message,
      value: defaultValue || '',
      position: 'center'
    }).then((result) => result?.value || null).catch(() => null);
  }

  // Legal AI specific dialogs
  function confirmCaseDelete(caseId: string): Promise<boolean> {
    return confirm(
      'Delete Case',
      `Are you sure you want to delete case ${caseId}? This action cannot be undone.`
    );
  }

  function confirmEvidenceDelete(evidenceId: string): Promise<boolean> {
    return confirm(
      'Delete Evidence',
      `Are you sure you want to delete evidence ${evidenceId}? This will remove all associated analysis.`
    );
  }

  function promptCaseName(): Promise<string | null> {
    return prompt(
      'Create New Case',
      'Enter a name for the new case:',
      'Untitled Case'
    );
  }

  function systemAlert(title: string, message: string): Promise<void> {
    return add({
      type: 'error',
      title,
      message,
      position: 'center',
      persistent: true
    });
  }

  return {
    subscribe,
    add,
    remove,
    reject,
    clear,
    info,
    success,
    warning,
    error,
    confirm,
    prompt,
    // Legal AI specific
    confirmCaseDelete,
    confirmEvidenceDelete,
    promptCaseName,
    systemAlert
  };
}

function createModalStore() {
  const { subscribe, update } = writable<Modal[]>([]);

  function generateId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  function add(modal: Omit<Modal, 'id'>): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = generateId();
      const newModal: Modal = {
        ...modal,
        id,
        resolve,
        reject
      };

      update(modals => [...modals, newModal]);
    });
  }

  function remove(id: string, result?: unknown) {
    update(modals => {
      const modal = modals.find(m => m.id === id);
      if (modal?.resolve) {
        modal.resolve(result);
      }
      return modals.filter(m => m.id !== id);
    });
  }

  function reject(id: string, reason?: unknown) {
    update(modals => {
      const modal = modals.find(m => m.id === id);
      if (modal?.reject) {
        modal.reject(reason);
      }
      return modals.filter(m => m.id !== id);
    });
  }

  function clear() {
    update(modals => {
      modals.forEach(modal => {
        if (modal.reject) {
          modal.reject('cleared');
        }
      });
      return [];
    });
  }

  // Convenience method for opening custom component modals
  function open(component: any, props: Record<string, any> = {}, options: Partial<Modal> = {}): Promise<any> {
    return add({
      component,
      props,
      size: options.size || 'md',
      type: options.type || 'default',
      persistent: options.persistent || false
    });
  }

  return {
    subscribe,
    add,
    remove,
    reject,
    clear,
    open
  };
}

export const dialogStore = createDialogStore();
export const modalStore = createModalStore();
;
// Export convenience functions
export const dialog = dialogStore;
export const modal = modalStore;