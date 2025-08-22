/**
 * Interaction Tracker Service
 * Advanced user behavior tracking for AI assistant context awareness
 * Implements: Real-time interaction logging, pattern analysis, predictive caching
 */

import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

export interface UserInteraction {
  type: 'mouse_move' | 'click' | 'hover' | 'scroll' | 'key_press' | 'document_click' | 'document_hover' | 'search' | 'upload';
  timestamp: number;
  position: { x: number; y: number };
  documentId?: string;
  target: HTMLElement | string;
  metadata?: Record<string, any>;
  sessionId?: string;
  userId?: string;
}

export interface InteractionPattern {
  pattern: string;
  frequency: number;
  avgDuration: number;
  lastOccurrence: number;
  predictedNext?: string[];
}

export interface UserContext {
  currentFocus: string | null;
  recentDocuments: string[];
  searchTerms: string[];
  workflowState: 'exploring' | 'analyzing' | 'writing' | 'reviewing';
  expertise: 'novice' | 'intermediate' | 'expert';
  preferences: {
    visualMode: '2d' | '3d' | 'hybrid';
    aiAssistance: 'minimal' | 'moderate' | 'extensive';
    shortcuts: Record<string, string>;
  };
}

class InteractionTrackerService {
  private interactions = writable<UserInteraction[]>([]);
  private patterns = writable<Map<string, InteractionPattern>>(new Map());
  private userContext = writable<UserContext>({
    currentFocus: null,
    recentDocuments: [],
    searchTerms: [],
    workflowState: 'exploring',
    expertise: 'intermediate',
    preferences: {
      visualMode: '3d',
      aiAssistance: 'moderate',
      shortcuts: {}
    }
  });

  private sessionId: string;
  private isTracking = false;
  private interactionBuffer: UserInteraction[] = [];
  private bufferSize = 100;
  private flushInterval = 5000; // 5 seconds
  private flushTimer: NodeJS.Timeout | null = null;

  // Hidden Markov Model for pattern prediction
  private transitionMatrix = new Map<string, Map<string, number>>();
  private stateSequence: string[] = [];

  constructor() {
    this.sessionId = this.generateSessionId();
    
    if (browser) {
      this.loadPersistedData();
      this.startTracking();
      this.startFlushTimer();
    }
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  startTracking() {
    if (this.isTracking || !browser) return;
    
    this.isTracking = true;
    this.setupEventListeners();
    console.log('🎯 Interaction tracking started');
  }

  stopTracking() {
    this.isTracking = false;
    this.removeEventListeners();
    this.flushBuffer();
    
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    
    console.log('🎯 Interaction tracking stopped');
  }

  private setupEventListeners() {
    if (!browser) return;

    // Mouse tracking
    document.addEventListener('mousemove', this.handleMouseMove, { passive: true });
    document.addEventListener('click', this.handleClick, { passive: true });
    
    // Keyboard tracking
    document.addEventListener('keydown', this.handleKeyPress, { passive: true });
    
    // Scroll tracking
    document.addEventListener('scroll', this.handleScroll, { passive: true });
    
    // Focus tracking
    document.addEventListener('focusin', this.handleFocus, { passive: true });
    document.addEventListener('focusout', this.handleBlur, { passive: true });

    // Page visibility for session management
    document.addEventListener('visibilitychange', this.handleVisibilityChange);
  }

  private removeEventListeners() {
    if (!browser) return;

    document.removeEventListener('mousemove', this.handleMouseMove);
    document.removeEventListener('click', this.handleClick);
    document.removeEventListener('keydown', this.handleKeyPress);
    document.removeEventListener('scroll', this.handleScroll);
    document.removeEventListener('focusin', this.handleFocus);
    document.removeEventListener('focusout', this.handleBlur);
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
  }

  private handleMouseMove = (event: MouseEvent) => {
    // Throttle mouse move events
    if (Date.now() % 100 === 0) { // Only record every ~100ms
      this.recordInteraction({
        type: 'mouse_move',
        timestamp: Date.now(),
        position: { x: event.clientX, y: event.clientY },
        target: event.target as HTMLElement,
        metadata: {
          buttons: event.buttons,
          ctrlKey: event.ctrlKey,
          altKey: event.altKey,
          shiftKey: event.shiftKey
        }
      });
    }
  };

  private handleClick = (event: MouseEvent) => {
    const target = event.target as HTMLElement;
    const documentId = this.extractDocumentId(target);
    
    this.recordInteraction({
      type: 'click',
      timestamp: Date.now(),
      position: { x: event.clientX, y: event.clientY },
      target,
      documentId,
      metadata: {
        button: event.button,
        ctrlKey: event.ctrlKey,
        altKey: event.altKey,
        shiftKey: event.shiftKey,
        targetTagName: target.tagName,
        targetId: target.id,
        targetClass: target.className
      }
    });

    // Update user context based on click
    this.updateUserContext('click', { documentId, target });
  };

  private handleKeyPress = (event: KeyboardEvent) => {
    this.recordInteraction({
      type: 'key_press',
      timestamp: Date.now(),
      position: { x: 0, y: 0 }, // Keyboard events don't have position
      target: event.target as HTMLElement,
      metadata: {
        key: event.key,
        code: event.code,
        ctrlKey: event.ctrlKey,
        altKey: event.altKey,
        shiftKey: event.shiftKey,
        repeat: event.repeat
      }
    });

    // Detect workflow patterns from key combinations
    this.analyzeKeyboardWorkflow(event);
  };

  private handleScroll = (event: Event) => {
    this.recordInteraction({
      type: 'scroll',
      timestamp: Date.now(),
      position: { x: window.scrollX, y: window.scrollY },
      target: event.target as HTMLElement,
      metadata: {
        scrollTop: document.documentElement.scrollTop,
        scrollLeft: document.documentElement.scrollLeft,
        direction: this.getScrollDirection()
      }
    });
  };

  private handleFocus = (event: FocusEvent) => {
    const target = event.target as HTMLElement;
    const documentId = this.extractDocumentId(target);
    
    this.userContext.update(ctx => ({
      ...ctx,
      currentFocus: documentId || target.id || target.tagName
    }));
  };

  private handleBlur = (event: FocusEvent) => {
    this.userContext.update(ctx => ({
      ...ctx,
      currentFocus: null
    }));
  };

  private handleVisibilityChange = () => {
    if (document.hidden) {
      this.flushBuffer();
    } else {
      // User returned, potentially trigger proactive loading
      this.triggerProactiveLoading();
    }
  };

  private extractDocumentId(element: HTMLElement): string | undefined {
    // Look for document ID in data attributes or closest parent
    let current = element;
    while (current && current !== document.body) {
      if (current.dataset.documentId) {
        return current.dataset.documentId;
      }
      if (current.id?.startsWith('doc-')) {
        return current.id.replace('doc-', '');
      }
      current = current.parentElement!;
    }
    return undefined;
  }

  recordInteraction(interaction: UserInteraction) {
    if (!this.isTracking) return;

    // Add session info
    interaction.sessionId = this.sessionId;
    interaction.userId = this.getCurrentUserId();

    // Add to buffer
    this.interactionBuffer.push(interaction);

    // Update pattern analysis
    this.updatePatterns(interaction);
    
    // Update HMM state sequence
    this.updateStateSequence(interaction);

    // Flush buffer if full
    if (this.interactionBuffer.length >= this.bufferSize) {
      this.flushBuffer();
    }
  }

  private updatePatterns(interaction: UserInteraction) {
    const patternKey = `${interaction.type}_${interaction.target}`;
    
    this.patterns.update(patterns => {
      const existing = patterns.get(patternKey);
      
      if (existing) {
        existing.frequency++;
        existing.lastOccurrence = interaction.timestamp;
        // Update average duration if applicable
      } else {
        patterns.set(patternKey, {
          pattern: patternKey,
          frequency: 1,
          avgDuration: 0,
          lastOccurrence: interaction.timestamp,
          predictedNext: this.predictNextActions(patternKey)
        });
      }
      
      return patterns;
    });
  }

  private updateStateSequence(interaction: UserInteraction) {
    const state = `${interaction.type}_${interaction.documentId ? 'doc' : 'ui'}`;
    
    this.stateSequence.push(state);
    
    // Keep only last 50 states for HMM
    if (this.stateSequence.length > 50) {
      this.stateSequence = this.stateSequence.slice(-50);
    }

    // Update transition matrix
    if (this.stateSequence.length >= 2) {
      const prevState = this.stateSequence[this.stateSequence.length - 2];
      const currentState = state;
      
      if (!this.transitionMatrix.has(prevState)) {
        this.transitionMatrix.set(prevState, new Map());
      }
      
      const transitions = this.transitionMatrix.get(prevState)!;
      transitions.set(currentState, (transitions.get(currentState) || 0) + 1);
    }
  }

  private predictNextActions(currentPattern: string): string[] {
    const predictions: string[] = [];
    
    // Use transition matrix to predict next likely actions
    const currentState = this.stateSequence[this.stateSequence.length - 1];
    const transitions = this.transitionMatrix.get(currentState);
    
    if (transitions) {
      const sorted = Array.from(transitions.entries())
        .sort(([,a], [,b]) => b - a)
        .slice(0, 3) // Top 3 predictions
        .map(([state]) => state);
      
      predictions.push(...sorted);
    }
    
    return predictions;
  }

  private updateUserContext(eventType: string, data: unknown) {
    this.userContext.update(ctx => {
      const newCtx = { ...ctx };
      
      // Update recent documents
      if (data.documentId && !newCtx.recentDocuments.includes(data.documentId)) {
        newCtx.recentDocuments = [data.documentId, ...newCtx.recentDocuments.slice(0, 9)];
      }
      
      // Infer workflow state from interaction patterns
      newCtx.workflowState = this.inferWorkflowState();
      
      return newCtx;
    });
  }

  private inferWorkflowState(): 'exploring' | 'analyzing' | 'writing' | 'reviewing' {
    const recentInteractions = this.interactionBuffer.slice(-20);
    
    const clickCount = recentInteractions.filter(i => i.type === 'click').length;
    const keyCount = recentInteractions.filter(i => i.type === 'key_press').length;
    const scrollCount = recentInteractions.filter(i => i.type === 'scroll').length;
    
    if (keyCount > clickCount * 2) return 'writing';
    if (scrollCount > clickCount) return 'reviewing';
    if (clickCount > 5) return 'exploring';
    
    return 'analyzing';
  }

  private analyzeKeyboardWorkflow(event: KeyboardEvent) {
    // Detect common keyboard shortcuts and workflow patterns
    const shortcuts = {
      'Ctrl+F': 'search',
      'Ctrl+S': 'save',
      'Ctrl+C': 'copy',
      'Ctrl+V': 'paste',
      'Ctrl+Z': 'undo',
      'Escape': 'cancel'
    };

    const combo = `${event.ctrlKey ? 'Ctrl+' : ''}${event.key}`;
    const action = shortcuts[combo];
    
    if (action) {
      // Update user context with detected action
      this.recordInteraction({
        type: 'key_press',
        timestamp: Date.now(),
        position: { x: 0, y: 0 },
        target: event.target as HTMLElement,
        metadata: {
          action,
          shortcut: combo
        }
      });
    }
  }

  private getScrollDirection(): 'up' | 'down' | 'left' | 'right' | 'none' {
    // Simple scroll direction detection
    // This could be enhanced with velocity and acceleration tracking
    return 'down'; // Placeholder
  }

  private getCurrentUserId(): string | undefined {
    // Retrieve from authentication store or localStorage
    return localStorage.getItem('userId') || undefined;
  }

  private startFlushTimer() {
    this.flushTimer = setInterval(() => {
      this.flushBuffer();
    }, this.flushInterval);
  }

  private flushBuffer() {
    if (this.interactionBuffer.length === 0) return;

    // Add to store
    this.interactions.update(current => [
      ...current.slice(-(1000 - this.interactionBuffer.length)), // Keep last 1000 interactions
      ...this.interactionBuffer
    ]);

    // Persist to IndexedDB
    this.persistInteractions(this.interactionBuffer);

    // Send to backend for analysis
    this.sendToBackend(this.interactionBuffer);

    // Clear buffer
    this.interactionBuffer = [];
  }

  private async persistInteractions(interactions: UserInteraction[]) {
    if (!browser) return;

    try {
      // Store in IndexedDB for offline access
      const db = await this.openDB();
      const tx = db.transaction(['interactions'], 'readwrite');
      const store = tx.objectStore('interactions');
      
      for (const interaction of interactions) {
        await store.add(interaction);
      }
      
      await tx.complete;
    } catch (error) {
      console.error('Failed to persist interactions:', error);
    }
  }

  private async sendToBackend(interactions: UserInteraction[]) {
    try {
      await fetch('/api/interactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: this.sessionId,
          interactions
        })
      });
    } catch (error) {
      console.error('Failed to send interactions to backend:', error);
    }
  }

  private async triggerProactiveLoading() {
    // Analyze recent patterns to predict what user might need
    const predictions = this.predictNextActions('current');
    
    if (predictions.length > 0) {
      // Trigger preloading of likely-needed resources
      fetch('/api/proactive-cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          predictions,
          userContext: this.userContext
        })
      }).catch(console.error);
    }
  }

  private async loadPersistedData() {
    try {
      const db = await this.openDB();
      const tx = db.transaction(['interactions'], 'readonly');
      const store = tx.objectStore('interactions');
      
      // Load recent interactions for pattern analysis
      const recentInteractions = await store.getAll();
      
      if (recentInteractions.length > 0) {
        this.interactions.set(recentInteractions.slice(-1000));
        
        // Rebuild patterns from persisted data
        for (const interaction of recentInteractions) {
          this.updatePatterns(interaction);
          this.updateStateSequence(interaction);
        }
      }
    } catch (error) {
      console.error('Failed to load persisted interactions:', error);
    }
  }

  private async openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('InteractionTrackerDB', 1);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
      
      request.onupgradeneeded = () => {
        const db = request.result;
        
        if (!db.objectStoreNames.contains('interactions')) {
          const store = db.createObjectStore('interactions', { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          store.createIndex('timestamp', 'timestamp');
          store.createIndex('type', 'type');
          store.createIndex('sessionId', 'sessionId');
        }
      };
    });
  }

  // Public API for components
  getInteractions() {
    return this.interactions;
  }

  getPatterns() {
    return this.patterns;
  }

  getUserContext() {
    return this.userContext;
  }

  // Derived stores for specific insights
  getRecentActivity() {
    return derived(this.interactions, $interactions => 
      $interactions.slice(-50).reverse()
    );
  }

  getWorkflowInsights() {
    return derived([this.interactions, this.patterns], ([$interactions, $patterns]) => ({
      totalInteractions: $interactions.length,
      averageSessionLength: this.calculateAverageSessionLength($interactions),
      mostUsedFeatures: Array.from($patterns.entries())
        .sort(([,a], [,b]) => b.frequency - a.frequency)
        .slice(0, 5),
      workflowEfficiency: this.calculateWorkflowEfficiency($interactions)
    }));
  }

  private calculateAverageSessionLength(interactions: UserInteraction[]): number {
    // Calculate average time between first and last interaction per session
    const sessions = new Map<string, UserInteraction[]>();
    
    for (const interaction of interactions) {
      if (!interaction.sessionId) continue;
      
      if (!sessions.has(interaction.sessionId)) {
        sessions.set(interaction.sessionId, []);
      }
      sessions.get(interaction.sessionId)!.push(interaction);
    }

    let totalDuration = 0;
    let sessionCount = 0;

    for (const [, sessionInteractions] of sessions) {
      if (sessionInteractions.length < 2) continue;
      
      const sorted = sessionInteractions.sort((a, b) => a.timestamp - b.timestamp);
      const duration = sorted[sorted.length - 1].timestamp - sorted[0].timestamp;
      
      totalDuration += duration;
      sessionCount++;
    }

    return sessionCount > 0 ? totalDuration / sessionCount : 0;
  }

  private calculateWorkflowEfficiency(interactions: UserInteraction[]): number {
    // Simple efficiency metric based on action-to-result ratio
    const actionInteractions = interactions.filter(i => 
      ['click', 'key_press'].includes(i.type)
    ).length;
    
    const resultInteractions = interactions.filter(i => 
      i.metadata?.action && ['save', 'submit', 'complete'].includes(i.metadata.action)
    ).length;

    return resultInteractions > 0 ? actionInteractions / resultInteractions : 0;
  }
}

// Singleton instance
export const interactionTracker = new InteractionTrackerService();