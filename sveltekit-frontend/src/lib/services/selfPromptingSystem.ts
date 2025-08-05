// Self-Prompting & Auto-Save System
// Intelligent user intent detection with contextual recommendations

import { writable } from 'svelte/store';
import { ChatOllama } from '@langchain/ollama';
import { crewAIOrchestrator } from '$lib/ai/crewai-legal-agents';
import { documentUpdateLoop } from '$lib/services/documentUpdateLoop';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface UserContext {
  // Document context
  documentId: string;
  documentType: 'contract' | 'brief' | 'memo' | 'agreement' | 'other';
  currentContent: string;
  contentLength: number;
  
  // User behavior
  lastActivity: Date;
  activityPattern: 'writing' | 'editing' | 'reviewing' | 'idle' | 'researching';
  keystrokes: number;
  idleDuration: number; // milliseconds
  
  // Session context
  sessionStart: Date;
  totalEdits: number;
  lastSaveTime: Date | null;
  unsavedChanges: boolean;
  
  // AI interaction history
  lastAIInteraction: Date | null;
  acceptedSuggestions: number;
  rejectedSuggestions: number;
  
  // Focus and intent
  currentFocus: 'content_creation' | 'legal_review' | 'formatting' | 'research';
  userGoal: string; // Inferred or explicitly set
  confidenceLevel: number; // How confident we are about user intent
}

export interface SelfPrompt {
  id: string;
  type: 'auto_save' | 'suggestion' | 'focus_change' | 'break_reminder' | 'progress_summary';
  message: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  actions: Array<{
    id: string;
    label: string;
    action: () => Promise<void>;
    primary?: boolean;
  }>;
  dismissible: boolean;
  autoHideAfter?: number; // milliseconds
  position?: 'top' | 'center' | 'bottom';
}

export interface ActivityMetrics {
  wordsPerMinute: number;
  editingVelocity: number; // edits per minute
  focusScore: number; // 0-1, how focused the user appears
  productivityScore: number; // 0-1, overall productivity
  suggestions: {
    breakRecommended: boolean;
    focusChangeRecommended: boolean;
    aiAssistanceRecommended: boolean;
  };
}

// ============================================================================
// SELF-PROMPTING SYSTEM
// ============================================================================

export class SelfPromptingSystem {
  private ollama: ChatOllama;
  private userContext: UserContext;
  private activityTimer: NodeJS.Timeout | null = null;
  private autoSaveTimer: NodeJS.Timeout | null = null;
  private contextAnalysisTimer: NodeJS.Timeout | null = null;
  
  // Event tracking
  private keystrokeBuffer: number[] = [];
  private editHistory: Array<{ timestamp: Date; type: string; content: string }> = [];
  
  constructor(initialContext: Partial<UserContext>) {
    this.ollama = new ChatOllama({
      baseUrl: 'http://localhost:11434',
      model: 'gemma3-legal',
    });
    
    this.userContext = {
      documentId: '',
      documentType: 'other',
      currentContent: '',
      contentLength: 0,
      lastActivity: new Date(),
      activityPattern: 'writing',
      keystrokes: 0,
      idleDuration: 0,
      sessionStart: new Date(),
      totalEdits: 0,
      lastSaveTime: null,
      unsavedChanges: false,
      lastAIInteraction: null,
      acceptedSuggestions: 0,
      rejectedSuggestions: 0,
      currentFocus: 'content_creation',
      userGoal: 'Create legal document',
      confidenceLevel: 0.5,
      ...initialContext
    };
    
    this.startActivityMonitoring();
  }

  // ============================================================================
  // ACTIVITY MONITORING
  // ============================================================================

  private startActivityMonitoring() {
    // Monitor activity every 30 seconds
    this.activityTimer = setInterval(() => {
      this.analyzeUserActivity();
    }, 30000);
    
    // Auto-save check every 5 minutes
    this.autoSaveTimer = setInterval(() => {
      this.checkAutoSave();
    }, 300000);
    
    // Deep context analysis every 2 minutes
    this.contextAnalysisTimer = setInterval(() => {
      this.performContextAnalysis();
    }, 120000);
  }

  public recordActivity(activityType: string, data: any = {}) {
    const now = new Date();
    this.userContext.lastActivity = now;
    this.userContext.idleDuration = 0;
    
    // Update activity pattern
    this.updateActivityPattern(activityType, data);
    
    // Record keystroke if typing
    if (activityType === 'keystroke') {
      this.keystrokeBuffer.push(now.getTime());
      this.userContext.keystrokes++;
      
      // Keep only last minute of keystrokes
      const oneMinuteAgo = now.getTime() - 60000;
      this.keystrokeBuffer = this.keystrokeBuffer.filter(ts => ts > oneMinuteAgo);
    }
    
    // Record content changes
    if (activityType === 'content_change') {
      this.editHistory.push({
        timestamp: now,
        type: 'edit',
        content: data.content || ''
      });
      
      this.userContext.currentContent = data.content || '';
      this.userContext.contentLength = this.userContext.currentContent.length;
      this.userContext.unsavedChanges = true;
      this.userContext.totalEdits++;
      
      // Keep only last 10 edits
      if (this.editHistory.length > 10) {
        this.editHistory = this.editHistory.slice(-10);
      }
    }
    
    // Trigger immediate analysis for significant events
    if (['content_change', 'selection_change', 'focus_change'].includes(activityType)) {
      this.triggerContextualPrompts();
    }
  }

  private updateActivityPattern(activityType: string, data: any) {
    const patterns = {
      'keystroke': 'writing',
      'content_change': 'editing',
      'selection_change': 'reviewing',
      'scroll': 'reviewing',
      'copy': 'researching',
      'paste': 'writing'
    };
    
    if (patterns[activityType as keyof typeof patterns]) {
      this.userContext.activityPattern = patterns[activityType as keyof typeof patterns] as any;
    }
  }

  private analyzeUserActivity() {
    const now = new Date();
    const timeSinceLastActivity = now.getTime() - this.userContext.lastActivity.getTime();
    
    this.userContext.idleDuration = timeSinceLastActivity;
    
    // Check for idle state
    if (timeSinceLastActivity > 300000) { // 5 minutes
      this.userContext.activityPattern = 'idle';
      this.handleIdleUser();
    }
    
    // Calculate metrics
    const metrics = this.calculateActivityMetrics();
    
    // Generate contextual prompts based on metrics
    this.generateContextualPrompts(metrics);
  }

  // ============================================================================
  // INTELLIGENT PROMPTING
  // ============================================================================

  private async triggerContextualPrompts() {
    // Immediate contextual analysis based on current activity
    const context = this.analyzeCurrentContext();
    
    if (context.confidence > 0.7) {
      const prompts = await this.generateSmartPrompts(context);
      
      for (const prompt of prompts) {
        if (prompt.priority === 'high' || prompt.priority === 'urgent') {
          this.showPrompt(prompt);
        }
      }
    }
  }

  private analyzeCurrentContext(): any {
    const content = this.userContext.currentContent;
    const words = content.split(/\s+/).length;
    const sentences = content.split(/[.!?]+/).length;
    const avgWordsPerSentence = words / Math.max(sentences, 1);
    
    // Analyze content structure
    const hasLegalTerms = this.detectLegalLanguage(content);
    const complexity = this.calculateContentComplexity(content);
    const completeness = this.assessContentCompleteness(content);
    
    return {
      contentAnalysis: {
        words,
        sentences,
        avgWordsPerSentence,
        hasLegalTerms,
        complexity,
        completeness
      },
      userBehavior: {
        writingSpeed: this.keystrokeBuffer.length,
        editFrequency: this.userContext.totalEdits / this.getSessionDurationMinutes(),
        focusLevel: this.calculateFocusLevel()
      },
      confidence: this.calculateContextConfidence()
    };
  }

  private async generateSmartPrompts(context: any): Promise<SelfPrompt[]> {
    const prompts: SelfPrompt[] = [];
    
    // Auto-save prompts
    if (this.userContext.unsavedChanges && this.shouldPromptAutoSave()) {
      prompts.push(this.createAutoSavePrompt());
    }
    
    // Writing assistance prompts
    if (context.contentAnalysis.complexity < 0.3 && context.contentAnalysis.words > 100) {
      prompts.push(await this.createWritingAssistancePrompt(context));
    }
    
    // Legal review prompts
    if (context.contentAnalysis.hasLegalTerms && context.contentAnalysis.completeness > 0.7) {
      prompts.push(this.createLegalReviewPrompt());
    }
    
    // Focus change prompts
    if (this.userContext.activityPattern === 'idle' && this.userContext.idleDuration > 180000) {
      prompts.push(this.createFocusChangePrompt());
    }
    
    // Break reminders
    if (this.getSessionDurationMinutes() > 60 && context.userBehavior.focusLevel < 0.5) {
      prompts.push(this.createBreakReminderPrompt());
    }
    
    return prompts.filter(p => p !== null);
  }

  // ============================================================================
  // PROMPT CREATION
  // ============================================================================

  private createAutoSavePrompt(): SelfPrompt {
    return {
      id: `auto_save_${Date.now()}`,
      type: 'auto_save',
      message: `You have unsaved changes. Save your progress?`,
      priority: 'medium',
      actions: [
        {
          id: 'save_now',
          label: 'Save Now',
          action: async () => {
            await this.performAutoSave();
            this.userContext.lastSaveTime = new Date();
            this.userContext.unsavedChanges = false;
          },
          primary: true
        },
        {
          id: 'save_later',
          label: 'Later',
          action: async () => {
            // Schedule reminder in 5 minutes
            setTimeout(() => {
              if (this.userContext.unsavedChanges) {
                this.showPrompt(this.createAutoSavePrompt());
              }
            }, 300000);
          }
        }
      ],
      dismissible: true,
      autoHideAfter: 10000,
      position: 'top'
    };
  }

  private async createWritingAssistancePrompt(context: any): Promise<SelfPrompt> {
    const suggestions = await this.generateWritingSuggestions(context);
    
    return {
      id: `writing_help_${Date.now()}`,
      type: 'suggestion',
      message: `I notice you're working on ${this.userContext.documentType} content. Would you like AI assistance?`,
      priority: 'medium',
      actions: [
        {
          id: 'get_suggestions',
          label: 'Get Suggestions',
          action: async () => {
            await this.applyWritingSuggestions(suggestions);
          },
          primary: true
        },
        {
          id: 'start_review',
          label: 'Start AI Review',
          action: async () => {
            await this.startCrewAIReview();
          }
        },
        {
          id: 'dismiss',
          label: 'Not Now',
          action: async () => {}
        }
      ],
      dismissible: true,
      position: 'center'
    };
  }

  private createLegalReviewPrompt(): SelfPrompt {
    return {
      id: `legal_review_${Date.now()}`,
      type: 'suggestion',
      message: `Your document appears ready for legal review. Run comprehensive analysis?`,
      priority: 'high',
      actions: [
        {
          id: 'full_review',
          label: 'Full CrewAI Review',
          action: async () => {
            await this.startCrewAIReview();
          },
          primary: true
        },
        {
          id: 'compliance_check',
          label: 'Compliance Check Only',
          action: async () => {
            await this.startCrewAIReview(['compliance_specialist']);
          }
        },
        {
          id: 'later',
          label: 'Review Later',
          action: async () => {}
        }
      ],
      dismissible: true,
      position: 'center'
    };
  }

  private createFocusChangePrompt(): SelfPrompt {
    const suggestions = this.generateFocusChangeSuggestions();
    
    return {
      id: `focus_change_${Date.now()}`,
      type: 'focus_change',
      message: `You've been idle for a while. Would you like to switch focus or get a summary?`,
      priority: 'medium',
      actions: [
        {
          id: 'summarize',
          label: 'Summarize Progress',
          action: async () => {
            await this.generateProgressSummary();
          },
          primary: true
        },
        {
          id: 'change_focus',
          label: 'Change Focus',
          action: async () => {
            this.userContext.currentFocus = suggestions.recommendedFocus;
          }
        },
        {
          id: 'continue',
          label: 'Continue Working',
          action: async () => {}
        }
      ],
      dismissible: true,
      autoHideAfter: 15000,
      position: 'center'
    };
  }

  private createBreakReminderPrompt(): SelfPrompt {
    const sessionDuration = this.getSessionDurationMinutes();
    
    return {
      id: `break_reminder_${Date.now()}`,
      type: 'break_reminder',
      message: `You've been working for ${sessionDuration} minutes. Consider taking a break?`,
      priority: 'low',
      actions: [
        {
          id: 'auto_save_break',
          label: 'Save & Break',
          action: async () => {
            await this.performAutoSave();
            this.showBreakModeUI();
          },
          primary: true
        },
        {
          id: 'continue',
          label: 'Keep Working',
          action: async () => {}
        }
      ],
      dismissible: true,
      autoHideAfter: 20000,
      position: 'bottom'
    };
  }

  // ============================================================================
  // CONTEXT ANALYSIS HELPERS
  // ============================================================================

  private detectLegalLanguage(content: string): boolean {
    const legalTerms = [
      'whereas', 'hereby', 'heretofore', 'pursuant', 'notwithstanding',
      'covenant', 'indemnify', 'liability', 'agreement', 'contract',
      'jurisdiction', 'breach', 'damages', 'remedy', 'enforce'
    ];
    
    const words = content.toLowerCase().split(/\s+/);
    const legalTermCount = words.filter(word => legalTerms.includes(word)).length;
    
    return legalTermCount > words.length * 0.02; // 2% threshold
  }

  private calculateContentComplexity(content: string): number {
    const sentences = content.split(/[.!?]+/);
    const avgLength = sentences.reduce((sum, s) => sum + s.split(/\s+/).length, 0) / sentences.length;
    
    // Normalize complexity score (0-1)
    return Math.min(avgLength / 30, 1);
  }

  private assessContentCompleteness(content: string): number {
    const indicators = [
      content.includes('WHEREAS'),
      content.includes('NOW THEREFORE'),
      content.includes('IN WITNESS WHEREOF'),
      content.length > 1000,
      content.split('\n\n').length > 3
    ];
    
    return indicators.filter(Boolean).length / indicators.length;
  }

  private calculateFocusLevel(): number {
    const recentEdits = this.editHistory.slice(-5);
    if (recentEdits.length < 2) return 0.5;
    
    // Calculate consistency in editing pattern
    const timeDiffs = [];
    for (let i = 1; i < recentEdits.length; i++) {
      const diff = recentEdits[i].timestamp.getTime() - recentEdits[i-1].timestamp.getTime();
      timeDiffs.push(diff);
    }
    
    const avgDiff = timeDiffs.reduce((sum, diff) => sum + diff, 0) / timeDiffs.length;
    const variance = timeDiffs.reduce((sum, diff) => sum + Math.pow(diff - avgDiff, 2), 0) / timeDiffs.length;
    
    // Lower variance = higher focus
    return Math.max(0, 1 - (variance / 1000000)); // Normalize
  }

  private calculateContextConfidence(): number {
    let confidence = 0.5;
    
    // Higher confidence with more activity
    if (this.userContext.totalEdits > 10) confidence += 0.2;
    if (this.userContext.keystrokes > 100) confidence += 0.2;
    
    // Lower confidence if user seems confused (lots of back-and-forth edits)
    const recentEditVariance = this.calculateEditVariance();
    if (recentEditVariance > 0.8) confidence -= 0.3;
    
    return Math.max(0, Math.min(1, confidence));
  }

  private calculateEditVariance(): number {
    if (this.editHistory.length < 3) return 0;
    
    const lengths = this.editHistory.map(edit => edit.content.length);
    const avg = lengths.reduce((sum, len) => sum + len, 0) / lengths.length;
    const variance = lengths.reduce((sum, len) => sum + Math.pow(len - avg, 2), 0) / lengths.length;
    
    return Math.min(variance / 10000, 1); // Normalize
  }

  private getSessionDurationMinutes(): number {
    return (Date.now() - this.userContext.sessionStart.getTime()) / 60000;
  }

  // ============================================================================
  // ACTION IMPLEMENTATIONS
  // ============================================================================

  private async performAutoSave(): Promise<void> {
    await documentUpdateLoop.queueDocumentUpdate(
      this.userContext.documentId,
      this.userContext.currentContent
    );
  }

  private async generateWritingSuggestions(context: any): Promise<string[]> {
    try {
      const response = await this.ollama.invoke([
        { role: 'system', content: 'You are a legal writing assistant. Provide 3 brief, specific suggestions to improve this legal document.' },
        { role: 'user', content: `Document type: ${this.userContext.documentType}\n\nContent: ${this.userContext.currentContent.substring(0, 1000)}...` }
      ]);
      
      return response.content.split('\n').filter(line => line.trim().length > 0).slice(0, 3);
    } catch {
      return ['Consider adding more specific legal language', 'Review for clarity and precision', 'Ensure all necessary clauses are included'];
    }
  }

  private async applyWritingSuggestions(suggestions: string[]): Promise<void> {
    // This would integrate with your suggestion application system
    console.log('Applying suggestions:', suggestions);
  }

  private async startCrewAIReview(agents?: string[]): Promise<void> {
    await crewAIOrchestrator.startDocumentReview({
      taskId: `self_prompt_${Date.now()}`,
      documentId: this.userContext.documentId,
      documentContent: this.userContext.currentContent,
      reviewType: 'comprehensive',
      priority: 'medium',
      assignedAgents: agents || ['compliance_specialist', 'risk_analyst', 'legal_editor']
    });
  }

  private generateFocusChangeSuggestions(): any {
    const currentFocus = this.userContext.currentFocus;
    const suggestions = {
      'content_creation': 'legal_review',
      'legal_review': 'formatting',
      'formatting': 'content_creation',
      'research': 'content_creation'
    };
    
    return {
      recommendedFocus: suggestions[currentFocus] || 'content_creation'
    };
  }

  private async generateProgressSummary(): Promise<void> {
    const summary = {
      wordsWritten: this.userContext.contentLength,
      editseMade: this.userContext.totalEdits,
      timeSpent: this.getSessionDurationMinutes(),
      completionEstimate: this.assessContentCompleteness(this.userContext.currentContent) * 100
    };
    
    this.showProgressSummary(summary);
  }

  private showBreakModeUI(): void {
    // This would trigger break mode in the UI
    console.log('Entering break mode');
  }

  private showProgressSummary(summary: any): void {
    // This would show progress summary in the UI
    console.log('Progress summary:', summary);
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private shouldPromptAutoSave(): boolean {
    if (!this.userContext.unsavedChanges) return false;
    if (!this.userContext.lastSaveTime) return this.userContext.totalEdits > 5;
    
    const timeSinceLastSave = Date.now() - this.userContext.lastSaveTime.getTime();
    return timeSinceLastSave > 300000; // 5 minutes
  }

  private calculateActivityMetrics(): ActivityMetrics {
    const sessionMinutes = this.getSessionDurationMinutes();
    
    return {
      wordsPerMinute: this.userContext.contentLength / Math.max(sessionMinutes, 1),
      editingVelocity: this.userContext.totalEdits / Math.max(sessionMinutes, 1),
      focusScore: this.calculateFocusLevel(),
      productivityScore: Math.min(1, (this.userContext.contentLength + this.userContext.totalEdits) / 1000),
      suggestions: {
        breakRecommended: sessionMinutes > 60,
        focusChangeRecommended: this.calculateFocusLevel() < 0.4,
        aiAssistanceRecommended: this.userContext.contentLength > 500 && this.userContext.totalEdits > 20
      }
    };
  }

  private async performContextAnalysis(): Promise<void> {
    const metrics = this.calculateActivityMetrics();
    const context = this.analyzeCurrentContext();
    
    // Generate proactive suggestions based on deep analysis
    if (context.confidence > 0.8) {
      const prompts = await this.generateSmartPrompts(context);
      
      // Show only high-priority prompts during context analysis
      prompts
        .filter(p => p.priority === 'high' || p.priority === 'urgent')
        .forEach(prompt => this.showPrompt(prompt));
    }
  }

  private checkAutoSave(): void {
    if (this.shouldPromptAutoSave()) {
      this.showPrompt(this.createAutoSavePrompt());
    }
  }

  // ============================================================================
  // PUBLIC API
  // ============================================================================

  public showPrompt(prompt: SelfPrompt): void {
    // This would integrate with your UI notification system
    console.log('Showing prompt:', prompt);
    selfPromptStore.update(prompts => [...prompts, prompt]);
    
    if (prompt.autoHideAfter) {
      setTimeout(() => {
        this.hidePrompt(prompt.id);
      }, prompt.autoHideAfter);
    }
  }

  public hidePrompt(promptId: string): void {
    selfPromptStore.update(prompts => prompts.filter(p => p.id !== promptId));
  }

  public updateContext(updates: Partial<UserContext>): void {
    this.userContext = { ...this.userContext, ...updates };
  }

  public getContext(): UserContext {
    return { ...this.userContext };
  }

  public destroy(): void {
    if (this.activityTimer) clearInterval(this.activityTimer);
    if (this.autoSaveTimer) clearInterval(this.autoSaveTimer);
    if (this.contextAnalysisTimer) clearInterval(this.contextAnalysisTimer);
  }
}

// ============================================================================
// STORES
// ============================================================================

export const selfPromptStore = writable<SelfPrompt[]>([]);
export const userContextStore = writable<UserContext | null>(null);

// ============================================================================
// SINGLETON FACTORY
// ============================================================================

let instance: SelfPromptingSystem | null = null;

export function createSelfPromptingSystem(initialContext: Partial<UserContext>): SelfPromptingSystem {
  if (instance) {
    instance.destroy();
  }
  
  instance = new SelfPromptingSystem(initialContext);
  userContextStore.set(instance.getContext());
  
  return instance;
}

export function getSelfPromptingSystem(): SelfPromptingSystem | null {
  return instance;
}