// Action directive type definitions for svelte-check compatibility
declare namespace svelteHTML {
  interface HTMLAttributes<T> {
    'use:melt'?: import('svelte/action').Action<T, any> | (() => void);
    'use:enhance'?: import('$app/forms').SubmitFunction;
    'use:draggable'?: import('svelte/action').Action<T, any>;
    'use:dndzone'?: import('svelte/action').Action<T, any>;
    'use:portal'?: import('svelte/action').Action<T, any>;
    'use:createEnhancedAction'?: import('svelte/action').Action<T, any>;
    'use:createEnhancedSubmit'?: import('svelte/action').Action<T, any>;
    'use:form'?: import('svelte/action').Action<T, any>;
    'use:formEnhance'?: import('svelte/action').Action<T, any>;
  }
}

declare module '@melt-ui/svelte' {
  export interface Action<T = HTMLElement, P = any> {
    (node: T, params?: P): {
      update?: (params: P) => void;
      destroy?: () => void;
    } | void;
  }

  // Builder return type for melt-ui
  export interface Builder<T = Record<string, any>> {
    subscribe(run: (value: T) => void): () => void;
    set?(value: T): void;
    update?(updater: (value: T) => T): void;
  }

  // Menu builder types
  export function createMenu(): {
    elements: {
      menu: Builder<Action>;
      item: Builder<Action>;
      trigger: Builder<Action>;
      separator: Builder<Action>;
    };
    states: {
      open: Builder<boolean>;
    };
  };

  // Dialog builder types
  export function createDialog(): {
    elements: {
      overlay: Builder<Action>;
      content: Builder<Action>;
      title: Builder<Action>;
      description: Builder<Action>;
      close: Builder<Action>;
      trigger: Builder<Action>;
    };
    states: {
      open: Builder<boolean>;
    };
  };

  // Toast builder types
  export function createToaster(): {
    elements: {
      content: (toast: any) => Builder<Action>;
      title: (toast: any) => Builder<Action>;
      description: (toast: any) => Builder<Action>;
      close: (toast: any) => Builder<Action>;
    };
    toasts: Builder<any[]>;
    helpers: {
      addToast: (data: any) => string;
    };
  };

  // Tabs builder types  
  export function createTabs(): {
    elements: {
      root: Builder<Action>;
      list: Builder<Action>;
      trigger: (value: string) => Builder<Action>;
      content: (value: string) => Builder<Action>;
    };
    states: {
      value: Builder<string>;
    };
  };

  // Common builders
  export function createAccordion(): any;
  export function createSelect(): any;
  export function createCombobox(): any;
  export function createDatePicker(): any;
}

// Svelte 5 action compatibility
declare module 'svelte/action' {
  export interface Action<Element = HTMLElement, Parameter = undefined> {
    <Node extends Element>(
      node: Node, 
      parameter?: Parameter
    ): {
      update?: (parameter: Parameter) => void;
      destroy?: () => void;
    } | void;
  }
}

// Legacy bits-ui compatibility (for existing components)
declare module 'bits-ui' {
  export interface Action<T = HTMLElement> {
    (node: T, params?: any): {
      update?: (params: any) => void;
      destroy?: () => void;
    };
  }

  export function createButton(): {
    elements: {
      root: any;
    };
    states: Record<string, any>;
    helpers: Record<string, any>;
  };
}

// Common action types for Svelte 5 compatibility
declare interface ActionStore<T> {
  subscribe: (fn: (value: T) => void) => () => void;
}