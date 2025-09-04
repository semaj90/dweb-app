/**
 * XState Idle Detection Service
 * Monitors user activity and triggers background processing during idle periods
 */

import { createMachine, interpret, type Actor } from 'xstate'
import { browser } from '$app/environment';

// Idle detection state machine
const idleMachine = createMachine({
  id: 'userIdle',
  initial: 'initializing',
  context: {
    idleThreshold: 300000, // 5 minutes in milliseconds
    lastActivity: Date.now(),
    backgroundTasks: [] as string[]
  },
  states: {
    initializing: {
      always: {
        target: 'active',
        actions: 'initializeActivity'
      }
    },
    active: {
      entry: 'recordActivity',
      after: {
        IDLE_TIMEOUT: {
          target: 'idle',
          guard: 'shouldGoIdle'
        }
      },
      on: {
        USER_ACTIVITY: {
          target: 'active',
          actions: 'recordActivity'
        },
        FORCE_IDLE: 'idle'
      }
    },
    idle: {
      entry: 'startBackgroundProcessing',
      exit: 'stopBackgroundProcessing',
      after: {
        BACKGROUND_CHECK: {
          target: 'idle',
          actions: 'performBackgroundTasks',
          internal: true
        }
      },
      on: {
        USER_ACTIVITY: {
          target: 'active',
          actions: 'recordActivity'
        }
      }
    }
  }
}, {
  delays: {
    IDLE_TIMEOUT: (context) => context.idleThreshold,
    BACKGROUND_CHECK: 10000 // Check every 10 seconds while idle
  },
  guards: {
    shouldGoIdle: (context) => {
      return Date.now() - context.lastActivity >= context.idleThreshold;
    }
  },
  actions: {
    initializeActivity: (context) => {
      context.lastActivity = Date.now();
    },
    recordActivity: (context) => {
      context.lastActivity = Date.now();
      console.log('🎯 User activity detected');
    },
    startBackgroundProcessing: (context) => {
      console.log('💤 User idle - starting background processing');
      // Trigger background cache hydration
      if (browser && typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('idle:start', {
          detail: { timestamp: Date.now() }
        }));
      }
    },
    stopBackgroundProcessing: (context) => {
      console.log('⚡ User active - stopping background processing');
      if (browser && typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('idle:stop', {
          detail: { timestamp: Date.now() }
        }));
      }
    },
    performBackgroundTasks: (context, event) => {
      console.log('🔄 Performing background maintenance tasks');
      if (browser && typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('idle:background-task', {
          detail: { 
            timestamp: Date.now(),
            tasks: context.backgroundTasks 
          }
        }));
      }
    }
  }
});

class IdleDetectionService {
  private actor: Actor<typeof idleMachine> | null = null;
  private activityListeners: (() => void)[] = [];
  private backgroundTasks: Set<() => Promise<void>> = new Set();

  constructor() {
    if (browser) {
      this.initializeService();
    }
  }

  private initializeService() {
    // Create and start the state machine
    this.actor = interpret(idleMachine);
    
    // Set up activity listeners
    this.setupActivityListeners();
    
    console.log('🎯 Idle detection service initialized');
  }

  private setupActivityListeners() {
    if (!browser || !this.actor) return;

    const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'];
    
    const activityHandler = () => {
      this.actor?.send({ type: 'USER_ACTIVITY' });
    };

    // Throttle activity detection to avoid excessive events
    let throttleTimer: number | null = null;
    const throttledHandler = () => {
      if (throttleTimer) return;
      
      throttleTimer = window.setTimeout(() => {
        activityHandler();
        throttleTimer = null;
      }, 1000); // Throttle to once per second
    };

    activityEvents.forEach(eventName => {
      document.addEventListener(eventName, throttledHandler, { passive: true });
      this.activityListeners.push(() => {
        document.removeEventListener(eventName, throttledHandler);
      });
    });

    // Clean up throttle timer when service is destroyed
    this.activityListeners.push(() => {
      if (throttleTimer) {
        clearTimeout(throttleTimer);
      }
    });
  }

  /**
   * Start the idle detection service
   */
  start() {
    if (!this.actor) {
      this.initializeService();
    }
    
    this.actor?.start();
    console.log('✅ Idle detection service started');
  }

  /**
   * Stop the idle detection service
   */
  stop() {
    if (this.actor) {
      this.actor.stop();
    }
    
    // Remove all activity listeners
    this.activityListeners.forEach(cleanup => cleanup());
    this.activityListeners = [];
    
    console.log('🛑 Idle detection service stopped');
  }

  /**
   * Get current state
   */
  get currentState() {
    return this.actor?.getSnapshot()?.value || 'unknown';
  }

  /**
   * Check if user is currently idle
   */
  get isIdle() {
    return this.currentState === 'idle';
  }

  /**
   * Check if user is active
   */
  get isActive() {
    return this.currentState === 'active';
  }

  /**
   * Get time since last activity
   */
  get timeSinceActivity() {
    const context = this.actor?.getSnapshot()?.context;
    return context ? Date.now() - context.lastActivity : 0;
  }

  /**
   * Force idle state (for testing or manual triggering)
   */
  forceIdle() {
    this.actor?.send({ type: 'FORCE_IDLE' });
  }

  /**
   * Manually trigger activity (for testing)
   */
  triggerActivity() {
    this.actor?.send({ type: 'USER_ACTIVITY' });
  }

  /**
   * Register a background task to run during idle periods
   */
  registerBackgroundTask(taskName: string, task: () => Promise<void>) {
    this.backgroundTasks.add(task);
    
    if (browser) {
      window.addEventListener('idle:background-task', async () => {
        try {
          console.log(`🔄 Running background task: ${taskName}`);
          await task();
        } catch (error) {
          console.error(`❌ Background task failed: ${taskName}`, error);
        }
      });
    }
  }

  /**
   * Unregister a background task
   */
  unregisterBackgroundTask(task: () => Promise<void>) {
    this.backgroundTasks.delete(task);
  }

  /**
   * Get service statistics
   */
  getStats() {
    const context = this.actor?.getSnapshot()?.context;
    return {
      currentState: this.currentState,
      isIdle: this.isIdle,
      timeSinceActivity: this.timeSinceActivity,
      lastActivity: context?.lastActivity || 0,
      idleThreshold: context?.idleThreshold || 0,
      backgroundTasksCount: this.backgroundTasks.size,
      activityListenersCount: this.activityListeners.length
    };
  }

  /**
   * Update idle threshold
   */
  setIdleThreshold(milliseconds: number) {
    if (this.actor) {
      // Update context
      const currentSnapshot = this.actor.getSnapshot();
      if (currentSnapshot.context) {
        currentSnapshot.context.idleThreshold = milliseconds;
      }
    }
  }
}

// Export singleton instance
export const idleService = new IdleDetectionService();
// Export the service class for testing
export { IdleDetectionService };