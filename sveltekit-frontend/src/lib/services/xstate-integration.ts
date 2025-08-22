/**
 * XState Integration Service - Complete Component Wiring
 * Connects all XState machines with Svelte components for comprehensive state management
 */

import { createActor, type ActorRefFrom } from 'xstate';
import { writable, derived, type Writable, type Readable } from 'svelte/store';

// Import all XState machines
import { authMachine, type AuthContext, type AuthEvent } from '$lib/machines/auth-machine.js';
import { sessionMachine, sessionServices, sessionActions } from '$lib/machines/sessionMachine.js';
import { aiAssistantMachine, type AIAssistantContext } from '$lib/machines/aiAssistantMachine.js';
import { agentShellMachine } from '$lib/machines/agentShellMachine.js';

// Import services
import { productionServiceClient, services } from './production-service-client.js';
import { mcpGPUOrchestrator } from './mcp-gpu-orchestrator.js';

// Global state interface
export interface GlobalAppState {
  auth: AuthContext;
  session: any;
  aiAssistant: any;
  agentShell: any;
  ui: {
    theme: 'light' | 'dark' | 'system';
    sidebarOpen: boolean;
    currentRoute: string;
    notifications: Notification[];
    isLoading: boolean;
  };
  legal: {
    activeCases: any[];
    currentCase: any;
    documents: any[];
    evidence: any[];
  };
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Date;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}

class XStateIntegrationService {
  // Actor instances
  public authActor: ActorRefFrom<typeof authMachine>;
  public sessionActor: ActorRefFrom<typeof sessionMachine>;
  public aiAssistantActor: ActorRefFrom<typeof aiAssistantMachine>;
  public agentShellActor: ActorRefFrom<typeof agentShellMachine>;

  // Svelte stores for reactive state
  public authState: Writable<AuthContext>;
  public sessionState: Writable<any>;
  public aiAssistantState: Writable<any>;
  public agentShellState: Writable<any>;
  public globalState: Writable<GlobalAppState>;

  // Derived stores for computed values
  public isAuthenticated: Readable<boolean>;
  public currentUser: Readable<any>;
  public hasPermission: Readable<(permission: string) => boolean>;
  public systemHealth: Readable<{
    auth: boolean;
    ai: boolean;
    services: boolean;
    overall: 'healthy' | 'degraded' | 'critical';
  }>;

  private subscriptions: (() => void)[] = [];

  constructor() {
    // Initialize actors with enhanced options
    this.authActor = createActor(authMachine, {
      input: {
        deviceInfo: this.getDeviceInfo(),
        timestamp: new Date().toISOString()
      }
    });

    this.sessionActor = createActor(sessionMachine.provide({
      services: sessionServices,
      actions: sessionActions
    }));

    this.aiAssistantActor = createActor(aiAssistantMachine);
    this.agentShellActor = createActor(agentShellMachine);

    // Initialize stores
    this.authState = writable(this.authActor.getSnapshot().context);
    this.sessionState = writable(this.sessionActor.getSnapshot().context);
    this.aiAssistantState = writable(this.aiAssistantActor.getSnapshot().context);
    this.agentShellState = writable(this.agentShellActor.getSnapshot().context);

    // Initialize global state
    this.globalState = writable({
      auth: this.authActor.getSnapshot().context,
      session: this.sessionActor.getSnapshot().context,
      aiAssistant: this.aiAssistantActor.getSnapshot().context,
      agentShell: this.agentShellActor.getSnapshot().context,
      ui: {
        theme: 'system',
        sidebarOpen: false,
        currentRoute: '/',
        notifications: [],
        isLoading: false
      },
      legal: {
        activeCases: [],
        currentCase: null,
        documents: [],
        evidence: []
      }
    } as GlobalAppState);

    // Create derived stores
    this.isAuthenticated = derived(
      this.authState,
      ($authState) => !!$authState.user && !!$authState.session
    );

    this.currentUser = derived(
      this.authState,
      ($authState) => $authState.user
    );

    this.hasPermission = derived(
      this.authState,
      ($authState) => (permission: string) => {
        return $authState.user?.permissions?.includes(permission) || 
               $authState.user?.permissions?.includes('all') ||
               false;
      }
    );

    this.systemHealth = derived(
      [this.authState, this.sessionState, this.aiAssistantState],
      ([$auth, $session, $aiAssistant]) => {
        const authHealthy = $auth.user !== null && !$auth.error;
        const sessionHealthy = $session.sessionHealth?.isValid !== false;
        const aiHealthy = $aiAssistant.ollamaClusterHealth?.primary !== false;
        
        const healthyCount = [authHealthy, sessionHealthy, aiHealthy].filter(Boolean).length;
        
        let overall: 'healthy' | 'degraded' | 'critical';
        if (healthyCount === 3) overall = 'healthy';
        else if (healthyCount >= 2) overall = 'degraded';
        else overall = 'critical';

        return {
          auth: authHealthy,
          ai: aiHealthy,
          services: sessionHealthy,
          overall
        };
      }
    );

    this.setupActorSubscriptions();
    this.startActors();
  }

  private setupActorSubscriptions(): void {
    // Auth actor subscription
    const authSub = this.authActor.subscribe((state) => {
      this.authState.set(state.context);
      this.globalState.update(global => ({
        ...global,
        auth: state.context
      }));

      // Handle authentication state changes
      if (state.matches('authenticated')) {
        this.onAuthenticationSuccess(state.context);
      } else if (state.matches('idle') && state.context.user === null) {
        this.onLogout();
      } else if (state.matches('error')) {
        this.showNotification({
          type: 'error',
          title: 'Authentication Error',
          message: state.context.error || 'Authentication failed'
        });
      }
    });

    // Session actor subscription
    const sessionSub = this.sessionActor.subscribe((state) => {
      this.sessionState.set(state.context);
      this.globalState.update(global => ({
        ...global,
        session: state.context
      }));

      // Handle session events
      if (state.matches('expired')) {
        this.authActor.send({ type: 'SESSION_EXPIRED' });
        this.showNotification({
          type: 'warning',
          title: 'Session Expired',
          message: 'Your session has expired. Please login again.'
        });
      }
    });

    // AI Assistant actor subscription
    const aiSub = this.aiAssistantActor.subscribe((state) => {
      this.aiAssistantState.set(state.context);
      this.globalState.update(global => ({
        ...global,
        aiAssistant: state.context
      }));

      // Handle AI responses
      if (state.context.response && state.context.response !== '') {
        // Could trigger UI updates, notifications, etc.
      }
    });

    // Agent Shell actor subscription
    const agentSub = this.agentShellActor.subscribe((state) => {
      this.agentShellState.set(state.context);
      this.globalState.update(global => ({
        ...global,
        agentShell: state.context
      }));
    });

    this.subscriptions.push(authSub, sessionSub, aiSub, agentSub);
  }

  private startActors(): void {
    this.authActor.start();
    this.sessionActor.start();
    this.aiAssistantActor.start();
    this.agentShellActor.start();
  }

  private async onAuthenticationSuccess(authContext: AuthContext): Promise<void> {
    // Start session management
    if (authContext.user && authContext.session) {
      this.sessionActor.send({
        type: 'AUTHENTICATE',
        user: authContext.user,
        sessionId: authContext.session.id || 'temp_session'
      });

      // Initialize AI assistant with user context
      this.aiAssistantActor.send({
        type: 'SET_MODEL',
        model: 'gemma3-legal'
      });

      // Check cluster health
      this.aiAssistantActor.send({ type: 'CHECK_CLUSTER_HEALTH' });

      // Show success notification
      this.showNotification({
        type: 'success',
        title: 'Welcome!',
        message: `Welcome back, ${authContext.user.firstName}!`
      });

      // Load user-specific data
      await this.loadUserData(authContext.user);
    }
  }

  private onLogout(): void {
    // Clear all state
    this.sessionActor.send({ type: 'LOGOUT' });
    this.aiAssistantActor.send({ type: 'CLEAR_CONVERSATION' });
    
    // Clear UI state
    this.globalState.update(global => ({
      ...global,
      legal: {
        activeCases: [],
        currentCase: null,
        documents: [],
        evidence: []
      },
      ui: {
        ...global.ui,
        notifications: []
      }
    }));

    this.showNotification({
      type: 'info',
      title: 'Logged Out',
      message: 'You have been successfully logged out.'
    });
  }

  private async loadUserData(user: any): Promise<void> {
    try {
      // Load user's active cases
      const casesResponse = await services.queryRAG('active cases', {
        userId: user.id,
        department: user.department,
        jurisdiction: user.jurisdiction
      });

      if (casesResponse.success) {
        this.globalState.update(global => ({
          ...global,
          legal: {
            ...global.legal,
            activeCases: casesResponse.data?.cases || []
          }
        }));
      }
    } catch (error) {
      console.error('Failed to load user data:', error);
    }
  }

  private getDeviceInfo() {
    if (typeof window === 'undefined') return {};
    
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      screenResolution: `${screen.width}x${screen.height}`,
      timestamp: new Date().toISOString()
    };
  }

  private showNotification(notification: Omit<Notification, 'id' | 'timestamp'>): void {
    const fullNotification: Notification = {
      ...notification,
      id: `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    };

    this.globalState.update(global => ({
      ...global,
      ui: {
        ...global.ui,
        notifications: [...global.ui.notifications, fullNotification]
      }
    }));

    // Auto-remove after 5 seconds for non-error notifications
    if (notification.type !== 'error') {
      setTimeout(() => {
        this.removeNotification(fullNotification.id);
      }, 5000);
    }
  }

  // Public API methods
  public login(email: string, password: string, options: {
    rememberMe?: boolean;
    twoFactorCode?: string;
  } = {}): void {
    this.authActor.send({
      type: 'START_LOGIN',
      data: {
        email,
        password,
        rememberMe: options.rememberMe,
        twoFactorCode: options.twoFactorCode,
        deviceInfo: this.getDeviceInfo()
      }
    });
  }

  public register(registrationData: any): void {
    this.authActor.send({
      type: 'START_REGISTRATION',
      data: {
        ...registrationData,
        deviceInfo: this.getDeviceInfo()
      }
    });
  }

  public logout(): void {
    this.authActor.send({ type: 'LOGOUT' });
  }

  public sendAIMessage(message: string, useContext7 = false): void {
    this.aiAssistantActor.send({
      type: 'SEND_MESSAGE',
      message,
      useContext7
    });
  }

  public analyzeWithContext7(topic: string): void {
    this.aiAssistantActor.send({
      type: 'ANALYZE_WITH_CONTEXT7',
      topic
    });
  }

  public setUITheme(theme: 'light' | 'dark' | 'system'): void {
    this.globalState.update(global => ({
      ...global,
      ui: {
        ...global.ui,
        theme
      }
    }));
  }

  public setSidebarOpen(open: boolean): void {
    this.globalState.update(global => ({
      ...global,
      ui: {
        ...global.ui,
        sidebarOpen: open
      }
    }));
  }

  public removeNotification(id: string): void {
    this.globalState.update(global => ({
      ...global,
      ui: {
        ...global.ui,
        notifications: global.ui.notifications.filter(n => n.id !== id)
      }
    }));
  }

  public recordActivity(route: string, action: string): void {
    this.sessionActor.send({
      type: 'ACTIVITY',
      route,
      action
    });
  }

  public checkPermission(permission: string): boolean {
    const authState = this.authActor.getSnapshot().context;
    return authState.user?.permissions?.includes(permission) ||
           authState.user?.permissions?.includes('all') ||
           false;
  }

  public async uploadDocument(file: File, metadata?: unknown): Promise<any> {
    try {
      const response = await productionServiceClient.execute('file.upload', {
        file,
        metadata: {
          ...metadata,
          uploadedBy: this.authActor.getSnapshot().context.user?.id,
          timestamp: new Date().toISOString()
        }
      });

      if (response.success) {
        this.showNotification({
          type: 'success',
          title: 'Upload Complete',
          message: `${file.name} has been uploaded successfully.`
        });

        // Refresh documents list
        await this.loadUserData(this.authActor.getSnapshot().context.user);
      }

      return response;
    } catch (error) {
      this.showNotification({
        type: 'error',
        title: 'Upload Failed',
        message: `Failed to upload ${file.name}: ${error}`
      });
      throw error;
    }
  }

  public destroy(): void {
    // Clean up subscriptions
    this.subscriptions.forEach(unsub => unsub());
    
    // Stop actors
    this.authActor.stop();
    this.sessionActor.stop();
    this.aiAssistantActor.stop();
    this.agentShellActor.stop();
  }
}

// Create singleton instance
export const xstateIntegration = new XStateIntegrationService();

// Export for use in components
export default xstateIntegration;

// Export types
export type { GlobalAppState, Notification };

// Export stores for direct component access
export const {
  authState,
  sessionState,
  aiAssistantState,
  agentShellState,
  globalState,
  isAuthenticated,
  currentUser,
  hasPermission,
  systemHealth
} = xstateIntegration;