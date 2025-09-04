import { writable } from "svelte/store";


export interface ModalConfig {
  id: string;
  component?: unknown; // Svelte component
  props?: Record<string, any>;
  title?: string;
  size?: "sm" | "md" | "lg" | "xl" | "full";
  closable?: boolean;
  persistent?: boolean; // Don't close on outside click
  onClose?: (...args: any[]) => void;
  onConfirm?: (...args: any[]) => void;
}

export interface ModalState {
  modals: ModalConfig[];
  activeModal: string | null;
}

const initialState: ModalState = {
  modals: [],
  activeModal: null,
};

function createModalStore() {
  const { subscribe, set, update } = writable<ModalState>(initialState);

  const store = {
    subscribe,

    // Open a modal
    open: (config: Omit<ModalConfig, "id"> & { id?: string }) => {
      const id =
        config.id ||
        `modal-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const modalConfig: ModalConfig = {
        size: "md",
        closable: true,
        persistent: false,
        ...config,
        id,
      };

      update((state) => ({
        modals: [...state.modals, modalConfig],
        activeModal: id,
      }));

      return id;
    },

    // Close a modal
    close: (id?: string) => {
      update((state) => {
        const modalToClose = id || state.activeModal;
        if (!modalToClose) return state;

        const modal = state.modals.find((m) => m.id === modalToClose);
        if (modal?.onClose) {
          modal.onClose();
        }
        const remainingModals = state.modals.filter(
          (m) => m.id !== modalToClose,
        );
        const newActiveModal =
          remainingModals.length > 0
            ? remainingModals[remainingModals.length - 1].id
            : null;

        return {
          modals: remainingModals,
          activeModal: newActiveModal,
        };
      });
    },

    // Close all modals
    closeAll: () => {
      update((state) => {
        // Call onClose for all modals
        state.modals.forEach((modal) => {
          if (modal.onClose) {
            modal.onClose();
          }
        });

        return initialState;
      });
    },

    // Update modal props
    updateProps: (id: string, props: Record<string, any>) => {
      update((state) => ({
        ...state,
        modals: state.modals.map((modal) =>
          modal.id === id
            ? { ...modal, props: { ...modal.props, ...props } }
            : modal,
        ),
      }));
    },

    // Convenience methods for common modal types
    confirm: (options: {
      title: string;
      message: string;
      confirmText?: string;
      cancelText?: string;
      onConfirm?: () => void;
      onCancel?: () => void;
    }) => {
      return store.open({
        title: options.title,
        component: "ConfirmModal", // This would be a built-in component
        props: {
          message: options.message,
          confirmText: options.confirmText || "Confirm",
          cancelText: options.cancelText || "Cancel",
        },
        size: "sm",
        onConfirm: options.onConfirm,
        onClose: options.onCancel,
      });
    },

    alert: (options: {
      title: string;
      message: string;
      buttonText?: string;
      onClose?: () => void;
    }) => {
      return store.open({
        title: options.title,
        component: "AlertModal", // This would be a built-in component
        props: {
          message: options.message,
          buttonText: options.buttonText || "OK",
        },
        size: "sm",
        onClose: options.onClose,
      });
    },

    prompt: (options: {
      title: string;
      message: string;
      placeholder?: string;
      defaultValue?: string;
      confirmText?: string;
      cancelText?: string;
      onConfirm?: (value: string) => void;
      onCancel?: () => void;
    }) => {
      return store.open({
        title: options.title,
        component: "PromptModal", // This would be a built-in component
        props: {
          message: options.message,
          placeholder: options.placeholder,
          defaultValue: options.defaultValue,
          confirmText: options.confirmText || "OK",
          cancelText: options.cancelText || "Cancel",
        },
        size: "sm",
        onConfirm: options.onConfirm,
        onClose: options.onCancel,
      });
    },
  };

  return store;
}

export const modals = createModalStore();
export default modals;
