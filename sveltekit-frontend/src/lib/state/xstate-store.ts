/// <reference types="vite/client" />

/**
 * XState Svelte Store Integration
 * Provides reactive Svelte stores for XState machines with persistence and devtools
 */
import { readable, writable, derived, type Readable, type Writable } from "svelte/store";
import {
  createActor,
  type ActorRef,
  type StateMachine,
  type AnyActorLogic
} from "xstate";
import { browser } from "$app/environment";
import {
  appMachine,
  appSelectors,
  type AppContext,
  type AppEvents
} from './app-machine';
import { legalCaseMachine, legalCaseSelectors } from './legal-case-machine';

// Store persistence interface
export interface StoreState {
  appState: any;
  legalCaseState: any;
  timestamp: number;
}

// Configuration for store behavior
export interface XStateStoreConfig {
  persist?: boolean;
  persistKey?: string;
  devtools?: boolean;
  logTransitions?: boolean;
  syncAcrossTabs?: boolean;
}

class XStateStoreManager {
  private static instance: XStateStoreManager;
  
  // Actor references
  private appActor: ActorRef<any> | null = null;
  private legalCaseActor: ActorRef<any> | null = null;
  
  // Svelte stores
  private appStore: Writable<any> | null = null;
  private legalCaseStore: Writable<any> | null = null;
  
  // Configuration
  private config: XStateStoreConfig;
  
  // State persistence
  private persistenceKey = 'xstate-store-state';
  private syncChannel: BroadcastChannel | null = null;

  private constructor(config: XStateStoreConfig = {}) {
    this.config = {
      persist: true,
      persistKey: 'legal-ai-state',
      devtools: browser && import.meta.env.NODE_ENV === 'development',
      logTransitions: browser && import.meta.env.NODE_ENV === 'development',
      syncAcrossTabs: true,
      ...config
    };
    
    if (browser) {
      this.initializeBrowserFeatures();
    }
  }

  public static getInstance(config?: XStateStoreConfig): XStateStoreManager {
    if (!XStateStoreManager.instance) {
      XStateStoreManager.instance = new XStateStoreManager(config);
    }
    return XStateStoreManager.instance;
  }

  /**
   * Initialize browser-specific features
   */
  private initializeBrowserFeatures(): void {
    // Set up cross-tab synchronization
    if (this.config.syncAcrossTabs) {
      this.syncChannel = new BroadcastChannel('xstate-sync');
      this.syncChannel.addEventListener('message', (event: any) => {
        this.handleCrossTabSync(event.data);
      });
    }

    // Listen for page unload to persist state
    if (this.config.persist) {
      window.addEventListener('beforeunload', () => {
        this.persistState();
      });

      // Periodic persistence (every 30 seconds)
      setInterval(() => {
        this.persistState();
      }, 30000);
    }

    // Set up online/offline detection
    window.addEventListener('online', () => {
      this.appActor?.send({ type: 'ONLINE' });
    });

    window.addEventListener('offline', () => {
      this.appActor?.send({ type: 'OFFLINE' });
    });

    // Set up performance monitoring
    this.setupPerformanceMonitoring();
  }

  /**
   * Initialize the application machine and store
   */
  public initializeApp(): {
    appStore: Readable<any>;
    appActor: ActorRef<any>;
    send: (event: AppEvents) => void;
    selectors: typeof appSelectors;
  } {
    if (this.appActor) {
      throw new Error('App machine already initialized');
    }

    // Load persisted state if available
    const persistedState = this.loadPersistedState();
    
    // Create app actor with persistence
    this.appActor = createActor(appMachine, {
      snapshot: persistedState?.appState,
      // Add devtools inspection
      inspect: this.config.devtools ? this.createDevtoolsInspector('app') : undefined
    });

    // Create reactive Svelte store
    const { subscribe } = readable(this.appActor.getSnapshot(), (set) => {
      // Subscribe to state changes
      const subscription = this.appActor!.subscribe((state) => {
        if (this.config.logTransitions) {
          console.log('🔄 App State Transition:', state.value, state.context);
        }
        
        set(state);
        
        // Broadcast to other tabs
        if (this.syncChannel) {
          this.syncChannel.postMessage({
            type: 'app-state-change',
            state: state
          });
        }
      });

      // Start the machine
      this.appActor!.start();

      // Cleanup subscription
      return () => {
        subscription.unsubscribe();
      };
    });

    // Send function for dispatching events
    const send = (event: AppEvents) => {
      if (this.config.logTransitions) {
        console.log('📤 App Event:', event);
      }
      this.appActor?.send(event);
    };

    return {
      appStore: { subscribe },
      appActor: this.appActor,
      send,
      selectors: appSelectors
    };
  }

  /**
   * Initialize the legal case machine and store
   */
  public initializeLegalCase(): {
    legalCaseStore: Readable<any>;
    legalCaseActor: ActorRef<any>;
    send: (event: any) => void;
    selectors: typeof legalCaseSelectors;
  } {
    if (this.legalCaseActor) {
      throw new Error('Legal case machine already initialized');
    }

    // Load persisted state if available
    const persistedState = this.loadPersistedState();
    
    // Create legal case actor
    this.legalCaseActor = createActor(legalCaseMachine, {
      snapshot: persistedState?.legalCaseState,
      inspect: this.config.devtools ? this.createDevtoolsInspector('legalCase') : undefined
    });

    // Create reactive Svelte store
    const { subscribe } = readable(this.legalCaseActor.getSnapshot(), (set) => {
      // Subscribe to state changes
      const subscription = this.legalCaseActor!.subscribe((state) => {
        if (this.config.logTransitions) {
          console.log('⚖️ Legal Case State Transition:', state.value, state.context);
        }
        
        set(state);
        
        // Broadcast to other tabs
        if (this.syncChannel) {
          this.syncChannel.postMessage({
            type: 'legal-case-state-change',
            state: state
          });
        }
      });

      // Start the machine
      this.legalCaseActor!.start();

      // Cleanup subscription
      return () => {
        subscription.unsubscribe();
      };
    });

    // Send function for dispatching events
    const send = (event: any) => {
      if (this.config.logTransitions) {
        console.log('📤 Legal Case Event:', event);
      }
      this.legalCaseActor?.send(event);
    };

    return {
      legalCaseStore: { subscribe },
      legalCaseActor: this.legalCaseActor,
      send,
      selectors: legalCaseSelectors
    };
  }

  /**
   * Create derived stores for specific state slices
   */
  public createDerivedStores(appStore: Readable<any>) {
    return {
      // User and authentication
      user: derived(appStore, ($app) => appSelectors.getCurrentUser($app)),
      isAuthenticated: derived(appStore, ($app) => appSelectors.isAuthenticated($app)),
      
      // UI state
      theme: derived(appStore, ($app) => appSelectors.getTheme($app)),
      layout: derived(appStore, ($app) => appSelectors.getLayout($app)),
      isGlobalLoading: derived(appStore, ($app) => appSelectors.isGlobalLoading($app)),
      loadingMessage: derived(appStore, ($app) => appSelectors.getLoadingMessage($app)),
      
      // Notifications
      notifications: derived(appStore, ($app) => appSelectors.getNotifications($app)),
      
      // Error handling
      error: derived(appStore, ($app) => appSelectors.getError($app)),
      hasError: derived(appStore, ($app) => appSelectors.hasError($app)),
      
      // Settings and features
      settings: derived(appStore, ($app) => appSelectors.getSettings($app)),
      features: derived(appStore, ($app) => appSelectors.getFeatures($app)),
      
      // Connection status
      isOnline: derived(appStore, ($app) => appSelectors.isOnline($app)),
      websocketStatus: derived(appStore, ($app) => appSelectors.getWebSocketStatus($app)),
      
      // Navigation
      currentRoute: derived(appStore, ($app) => appSelectors.getCurrentRoute($app)),
      breadcrumbs: derived(appStore, ($app) => appSelectors.getBreadcrumbs($app))
    };
  }

  /**
   * Create utility functions for state management
   */
  public createUtilities(appSend: (event: AppEvents) => void) {
    return {
      // Notification helpers
      notify: {
        success: (title: string, message: string) => 
          appSend({ type: 'ADD_NOTIFICATION', notification: { type: 'success', title, message } }),
        
        error: (title: string, message: string) => 
          appSend({ type: 'ADD_NOTIFICATION', notification: { type: 'error', title, message } }),
        
        warning: (title: string, message: string) => 
          appSend({ type: 'ADD_NOTIFICATION', notification: { type: 'warning', title, message } }),
        
        info: (title: string, message: string) => 
          appSend({ type: 'ADD_NOTIFICATION', notification: { type: 'info', title, message } }),
        
        dismiss: (id: string) => 
          appSend({ type: 'DISMISS_NOTIFICATION', id })
      },

      // Theme helpers
      theme: {
        setLight: () => appSend({ type: 'SET_THEME', theme: 'light' }),
        setDark: () => appSend({ type: 'SET_THEME', theme: 'dark' }),
        setAuto: () => appSend({ type: 'SET_THEME', theme: 'auto' })
      },

      // Layout helpers
      layout: {
        setDesktop: () => appSend({ type: 'SET_LAYOUT', layout: 'desktop' }),
        setTablet: () => appSend({ type: 'SET_LAYOUT', layout: 'tablet' }),
        setMobile: () => appSend({ type: 'SET_LAYOUT', layout: 'mobile' })
      },

      // Error helpers
      error: {
        set: (error: AppContext['error']) => appSend({ type: 'SET_ERROR', error }),
        clear: () => appSend({ type: 'CLEAR_ERROR' }),
        retry: () => appSend({ type: 'RETRY_FAILED_ACTION' })
      },

      // Loading helpers
      loading: {
        start: (message?: string) => appSend({ type: 'GLOBAL_LOADING', message }),
        stop: () => appSend({ type: 'GLOBAL_LOADING_COMPLETE' })
      },

      // Navigation helpers
      navigate: (path: string, title?: string) => 
        appSend({ type: 'NAVIGATE', path, title }),

      // Settings helpers
      settings: {
        update: (settings: Partial<AppContext['settings']>) => 
          appSend({ type: 'UPDATE_SETTINGS', settings }),
        reset: () => appSend({ type: 'RESET_SETTINGS' })
      }
    };
  }

  // Private helper methods

  private createDevtoolsInspector(machineId: string) {
    return (inspectionEvent: any) => {
      if (typeof window !== 'undefined' && (window as any).__REDUX_DEVTOOLS_EXTENSION__) {
        const devtools = (window as any).__REDUX_DEVTOOLS_EXTENSION__.connect({
          name: `XState: ${machineId}`,
          trace: true
        });

        switch (inspectionEvent.type) {
          case '@xstate.event':
            devtools.send(inspectionEvent.event, inspectionEvent.snapshot);
            break;
          case '@xstate.snapshot':
            devtools.init(inspectionEvent.snapshot);
            break;
        }
      }
    };
  }

  private persistState(): void {
    if (!this.config.persist || !browser) return;

    try {
      const state: StoreState = {
        appState: this.appActor?.getSnapshot(),
        legalCaseState: this.legalCaseActor?.getSnapshot(),
        timestamp: Date.now()
      };

      localStorage.setItem(this.config.persistKey!, JSON.stringify(state));
    } catch (error: any) {
      console.warn('Failed to persist XState store:', error);
    }
  }

  private loadPersistedState(): StoreState | null {
    if (!this.config.persist || !browser) return null;

    try {
      const stored = localStorage.getItem(this.config.persistKey!);
      if (!stored) return null;

      const state: StoreState = JSON.parse(stored);
      
      // Check if state is not too old (24 hours)
      const maxAge = 24 * 60 * 60 * 1000;
      if (Date.now() - state.timestamp > maxAge) {
        localStorage.removeItem(this.config.persistKey!);
        return null;
      }

      return state;
    } catch (error: any) {
      console.warn('Failed to load persisted XState store:', error);
      return null;
    }
  }

  private handleCrossTabSync(data: any): void {
    // Handle synchronization between tabs
    switch (data.type) {
      case 'app-state-change':
        // Update local state if needed
        break;
      case 'legal-case-state-change':
        // Update local state if needed
        break;
    }
  }

  private setupPerformanceMonitoring(): void {
    // Monitor page load performance
    if ('performance' in window) {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'navigation') {
            const navEntry = entry as PerformanceNavigationTiming;
            this.appActor?.send({
              type: 'UPDATE_PERFORMANCE_METRICS',
              metrics: {
                pageLoadTime: navEntry.loadEventEnd - navEntry.loadEventStart
              }
            });
          }
        }
      });

      observer.observe({ entryTypes: ['navigation'] });

      // Monitor memory usage if available
      if ('memory' in performance) {
        setInterval(() => {
          const memory = (performance as any).memory;
          this.appActor?.send({
            type: 'UPDATE_PERFORMANCE_METRICS',
            metrics: {
              memoryUsage: memory.usedJSHeapSize
            }
          });
        }, 30000); // Every 30 seconds
      }
    }
  }

  /**
   * Clean up resources
   */
  public destroy(): void {
    this.appActor?.stop();
    this.legalCaseActor?.stop();
    this.syncChannel?.close();
    
    if (browser && this.config.persist) {
      this.persistState();
    }
  }
}

// Export singleton instance and factory function
export const xstateStore = XStateStoreManager.getInstance();
;
// Factory function for creating custom store configurations
export function createXStateStore(config?: XStateStoreConfig) {
  return XStateStoreManager.getInstance(config);
}

// Convenience function for initializing all stores
export function initializeStores(config?: XStateStoreConfig) {
  const storeManager = createXStateStore(config);
  
  const { appStore, appActor, send: appSend, selectors: appSelectors } = storeManager.initializeApp();
  const derivedStores = storeManager.createDerivedStores(appStore);
  const utilities = storeManager.createUtilities(appSend);

  return {
    // Main stores
    appStore,
    appActor,
    
    // Derived stores
    ...derivedStores,
    
    // Event senders
    appSend,
    
    // Selectors
    appSelectors,
    
    // Utilities
    ...utilities,
    
    // Store manager for advanced usage
    storeManager
  };
}

// Type exports for better TypeScript support
export type XStateStores = ReturnType<typeof initializeStores>;
export type AppStoreState = ReturnType<typeof appSelectors.getCurrentUser>;

// Hook for Svelte components
export function useXStateStore() {
  return initializeStores();
}