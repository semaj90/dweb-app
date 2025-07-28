/**
 * Professional Keyboard Shortcuts Manager
 * Provides centralized keyboard shortcut handling for the legal AI platform
 */

export interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  alt?: boolean;
  shift?: boolean;
  meta?: boolean;
  preventDefault?: boolean;
  description: string;
  action: () => void;
}

export interface ShortcutGroup {
  name: string;
  shortcuts: ShortcutConfig[];
}

export class KeyboardShortcutManager {
  private shortcuts: Map<string, ShortcutConfig> = new Map();
  private groups: ShortcutGroup[] = [];
  private isListening = false;

  constructor() {
    this.handleKeydown = this.handleKeydown.bind(this);
  }

  /**
   * Register a keyboard shortcut
   */
  register(config: ShortcutConfig): void {
    const key = this.generateKey(config);
    this.shortcuts.set(key, config);
  }

  /**
   * Register multiple shortcuts as a group
   */
  registerGroup(group: ShortcutGroup): void {
    this.groups.push(group);
    group.shortcuts.forEach(shortcut => this.register(shortcut));
  }

  /**
   * Start listening for keyboard events
   */
  startListening(): void {
    if (!this.isListening) {
      document.addEventListener('keydown', this.handleKeydown);
      this.isListening = true;
    }
  }

  /**
   * Stop listening for keyboard events
   */
  stopListening(): void {
    if (this.isListening) {
      document.removeEventListener('keydown', this.handleKeydown);
      this.isListening = false;
    }
  }

  /**
   * Get all registered shortcuts organized by groups
   */
  getShortcuts(): ShortcutGroup[] {
    return this.groups;
  }

  /**
   * Handle keydown events
   */
  private handleKeydown(event: KeyboardEvent): void {
    const key = this.generateKeyFromEvent(event);
    const shortcut = this.shortcuts.get(key);

    if (shortcut) {
      if (shortcut.preventDefault !== false) {
        event.preventDefault();
      }
      shortcut.action();
    }
  }

  /**
   * Generate a unique key for a shortcut configuration
   */
  private generateKey(config: ShortcutConfig): string {
    const parts: string[] = [];
    
    if (config.ctrl) parts.push('ctrl');
    if (config.alt) parts.push('alt');
    if (config.shift) parts.push('shift');
    if (config.meta) parts.push('meta');
    
    parts.push(config.key.toLowerCase());
    
    return parts.join('+');
  }

  /**
   * Generate a key from a keyboard event
   */
  private generateKeyFromEvent(event: KeyboardEvent): string {
    const parts: string[] = [];
    
    if (event.ctrlKey) parts.push('ctrl');
    if (event.altKey) parts.push('alt');
    if (event.shiftKey) parts.push('shift');
    if (event.metaKey) parts.push('meta');
    
    parts.push(event.key.toLowerCase());
    
    return parts.join('+');
  }

  /**
   * Format shortcut for display
   */
  static formatShortcut(config: ShortcutConfig): string {
    const parts: string[] = [];
    
    if (config.ctrl) parts.push('Ctrl');
    if (config.alt) parts.push('Alt');
    if (config.shift) parts.push('Shift');
    if (config.meta) parts.push('Cmd');
    
    parts.push(config.key.toUpperCase());
    
    return parts.join(' + ');
  }
}

/**
 * Common keyboard shortcuts for legal applications
 */
export const commonShortcuts = {
  // Document operations
  save: { key: 's', ctrl: true, description: 'Save document' },
  saveAs: { key: 's', ctrl: true, shift: true, description: 'Save document as...' },
  newDocument: { key: 'n', ctrl: true, description: 'New document' },
  openDocument: { key: 'o', ctrl: true, description: 'Open document' },
  print: { key: 'p', ctrl: true, description: 'Print document' },
  
  // Editing
  undo: { key: 'z', ctrl: true, description: 'Undo' },
  redo: { key: 'y', ctrl: true, description: 'Redo' },
  cut: { key: 'x', ctrl: true, description: 'Cut' },
  copy: { key: 'c', ctrl: true, description: 'Copy' },
  paste: { key: 'v', ctrl: true, description: 'Paste' },
  selectAll: { key: 'a', ctrl: true, description: 'Select all' },
  
  // Formatting
  bold: { key: 'b', ctrl: true, description: 'Bold' },
  italic: { key: 'i', ctrl: true, description: 'Italic' },
  underline: { key: 'u', ctrl: true, description: 'Underline' },
  
  // Navigation
  find: { key: 'f', ctrl: true, description: 'Find' },
  findReplace: { key: 'h', ctrl: true, description: 'Find and replace' },
  goToLine: { key: 'g', ctrl: true, description: 'Go to line' },
  
  // View
  fullscreen: { key: 'F11', description: 'Toggle fullscreen' },
  focusMode: { key: 'F10', description: 'Toggle focus mode' },
  showShortcuts: { key: '/', ctrl: true, description: 'Show keyboard shortcuts' },
  
  // Legal-specific
  insertCitation: { key: 'k', ctrl: true, shift: true, description: 'Insert citation' },
  addEvidence: { key: 'e', ctrl: true, shift: true, description: 'Add evidence' },
  createNote: { key: 'm', ctrl: true, shift: true, description: 'Create note' },
  searchCaselaw: { key: 'l', ctrl: true, shift: true, description: 'Search case law' },
};

/**
 * Create a shortcut manager instance with common legal shortcuts
 */
export function createLegalShortcutManager(): KeyboardShortcutManager {
  const manager = new KeyboardShortcutManager();
  
  // Document Management Group
  manager.registerGroup({
    name: 'Document Management',
    shortcuts: [
      { ...commonShortcuts.save, action: () => console.log('Save triggered') },
      { ...commonShortcuts.saveAs, action: () => console.log('Save As triggered') },
      { ...commonShortcuts.newDocument, action: () => console.log('New Document triggered') },
      { ...commonShortcuts.openDocument, action: () => console.log('Open Document triggered') },
      { ...commonShortcuts.print, action: () => console.log('Print triggered') },
    ]
  });
  
  // Editing Group
  manager.registerGroup({
    name: 'Editing',
    shortcuts: [
      { ...commonShortcuts.undo, action: () => document.execCommand('undo') },
      { ...commonShortcuts.redo, action: () => document.execCommand('redo') },
      { ...commonShortcuts.cut, action: () => document.execCommand('cut') },
      { ...commonShortcuts.copy, action: () => document.execCommand('copy') },
      { ...commonShortcuts.paste, action: () => document.execCommand('paste') },
      { ...commonShortcuts.selectAll, action: () => document.execCommand('selectAll') },
    ]
  });
  
  // Formatting Group
  manager.registerGroup({
    name: 'Formatting',
    shortcuts: [
      { ...commonShortcuts.bold, action: () => document.execCommand('bold') },
      { ...commonShortcuts.italic, action: () => document.execCommand('italic') },
      { ...commonShortcuts.underline, action: () => document.execCommand('underline') },
    ]
  });
  
  // Legal Operations Group
  manager.registerGroup({
    name: 'Legal Operations',
    shortcuts: [
      { ...commonShortcuts.insertCitation, action: () => console.log('Insert Citation triggered') },
      { ...commonShortcuts.addEvidence, action: () => console.log('Add Evidence triggered') },
      { ...commonShortcuts.createNote, action: () => console.log('Create Note triggered') },
      { ...commonShortcuts.searchCaselaw, action: () => console.log('Search Case Law triggered') },
    ]
  });
  
  return manager;
}

/**
 * Hook for using keyboard shortcuts in Svelte components
 */
export function useKeyboardShortcuts(shortcuts: ShortcutConfig[]) {
  const manager = new KeyboardShortcutManager();
  
  shortcuts.forEach(shortcut => manager.register(shortcut));
  
  return {
    start: () => manager.startListening(),
    stop: () => manager.stopListening(),
    getShortcuts: () => manager.getShortcuts(),
  };
}

/**
 * Detect if user is on Mac for displaying correct modifier keys
 */
export function isMac(): boolean {
  return typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform);
}

/**
 * Get the correct modifier key text for the platform
 */
export function getModifierKey(): string {
  return isMac() ? '⌘' : 'Ctrl';
}

/**
 * Format shortcut text for display with platform-specific modifiers
 */
export function formatShortcutDisplay(config: ShortcutConfig): string {
  const parts: string[] = [];
  
  if (config.ctrl) parts.push(getModifierKey());
  if (config.alt) parts.push(isMac() ? '⌥' : 'Alt');
  if (config.shift) parts.push('⇧');
  if (config.meta) parts.push('⌘');
  
  // Handle special keys
  const keyMap: Record<string, string> = {
    'F11': 'F11',
    'F10': 'F10',
    'F9': 'F9',
    'F8': 'F8',
    'F7': 'F7',
    'F6': 'F6',
    'F5': 'F5',
    'F4': 'F4',
    'F3': 'F3',
    'F2': 'F2',
    'F1': 'F1',
    'escape': 'Esc',
    'enter': '↵',
    'tab': '⇥',
    'backspace': '⌫',
    'delete': '⌦',
    'arrowup': '↑',
    'arrowdown': '↓',
    'arrowleft': '←',
    'arrowright': '→',
    ' ': 'Space',
  };
  
  const displayKey = keyMap[config.key.toLowerCase()] || config.key.toUpperCase();
  parts.push(displayKey);
  
  return parts.join(' + ');
}

/**
 * Auto-save manager for documents
 */
export class AutoSaveManager {
  private interval: NodeJS.Timeout | null = null;
  private saveCallback: () => void;
  private intervalMs: number;
  private isDirty = false;

  constructor(saveCallback: () => void, intervalMs = 5000) {
    this.saveCallback = saveCallback;
    this.intervalMs = intervalMs;
  }

  start(): void {
    if (this.interval) return;
    
    this.interval = setInterval(() => {
      if (this.isDirty) {
        this.saveCallback();
        this.isDirty = false;
      }
    }, this.intervalMs);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  markDirty(): void {
    this.isDirty = true;
  }

  forceSave(): void {
    this.saveCallback();
    this.isDirty = false;
  }
}