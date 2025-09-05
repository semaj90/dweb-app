import crypto from "crypto";

import { writable, type Writable } from "svelte/store";
import { HistoryManager } from "./HistoryManager";

// Simple content type for now - will expand when Slate is properly integrated
export interface ContentNode {
  type: string;
  text?: string;
  children?: ContentNode[];
  [key: string]: unknown;
}

/**
 * Report class representing a document in the interactive canvas
 * Manages content, position, state, and history for undo/redo functionality
 */
export class Report {
  public id: string;
  public title: Writable<string>;
  public content: Writable<ContentNode[]>;
  public position: Writable<{ x: number; y: number }>;
  public size: Writable<{ width: number; height: number }>;
  public isDirty: Writable<boolean>;
  public version: Writable<number>;
  public historyManager: HistoryManager;

  // Metadata
  public caseId?: string;
  public createdAt: Date;
  public updatedAt: Date;
  public createdBy: string;
  public lastModifiedBy?: string;

  constructor(data: {
    id?: string;
    title?: string;
    content?: ContentNode[];
    posX?: number;
    posY?: number;
    width?: number;
    height?: number;
    caseId?: string;
    version?: number;
    createdBy: string;
  }) {
    this.id = data.id || crypto.randomUUID();
    this.title = writable(data.title || "Untitled Report");

    // Initialize with default content if none provided
    const initialContent: ContentNode[] = data.content || [
      {
        type: "paragraph",
        children: [{ type: "text", text: "" }],
      },
    ];

    this.content = writable(initialContent);
    this.position = writable({ x: data.posX || 50, y: data.posY || 50 });
    this.size = writable({
      width: data.width || 650,
      height: data.height || 450,
    });
    this.isDirty = writable(false);
    this.version = writable(data.version || 1);

    this.caseId = data.caseId;
    this.createdAt = new Date();
    this.updatedAt = new Date();
    this.createdBy = data.createdBy;

    // Initialize history manager
    this.historyManager = new HistoryManager(initialContent);
  }
  /**
   * Update the report content
   */
  updateContent(newContent: ContentNode[]): void {
    this.content.set(newContent);
    this.historyManager.addSnapshot(newContent);
    this.markDirty();
  }
  /**
   * Update the report title
   */
  updateTitle(newTitle: string): void {
    this.title.set(newTitle);
    this.markDirty();
  }
  /**
   * Update the report position
   */
  updatePosition(x: number, y: number): void {
    this.position.set({ x, y });
    this.markDirty();
  }
  /**
   * Update the report size
   */
  updateSize(width: number, height: number): void {
    this.size.set({ width, height });
    this.markDirty();
  }
  /**
   * Mark the report as dirty (needs saving)
   */
  markDirty(): void {
    this.isDirty.set(true);
    this.updatedAt = new Date();
  }
  /**
   * Mark the report as clean (saved)
   */
  markClean(): void {
    this.isDirty.set(false);
    this.version.update((v) => v + 1);
  }
  /**
   * Undo last change
   */
  undo(): boolean {
    const previousContent = this.historyManager.undo();
    if (previousContent) {
      this.content.set(previousContent);
      this.markDirty();
      return true;
    }
    return false;
  }
  /**
   * Redo last undone change
   */
  redo(): boolean {
    const nextContent = this.historyManager.redo();
    if (nextContent) {
      this.content.set(nextContent);
      this.markDirty();
      return true;
    }
    return false;
  }
  /**
   * Check if undo is available
   */
  canUndo(): boolean {
    return this.historyManager.canUndo();
  }
  /**
   * Check if redo is available
   */
  canRedo(): boolean {
    return this.historyManager.canRedo();
  }
  /**
   * Get serializable data for persistence
   */
  toJSON() {
    let currentTitle = "";
    let currentContent: ContentNode[] = [];
    let currentPosition = { x: 0, y: 0 };
    let currentSize = { width: 0, height: 0 };
    let currentVersion = 0;

    // Get current values from stores
    this.title.subscribe((value) => (currentTitle = value))();
    this.content.subscribe((value) => (currentContent = value))();
    this.position.subscribe((value) => (currentPosition = value))();
    this.size.subscribe((value) => (currentSize = value))();
    this.version.subscribe((value) => (currentVersion = value))();

    return {
      id: this.id,
      title: currentTitle,
      content: currentContent,
      posX: currentPosition.x,
      posY: currentPosition.y,
      width: currentSize.width,
      height: currentSize.height,
      version: currentVersion,
      caseId: this.caseId,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      createdBy: this.createdBy,
      lastModifiedBy: this.lastModifiedBy,
    };
  }
  /**
   * Create a Report instance from database data
   */
  static fromJSON(data: any, createdBy: string): Report {
    return new Report({
      id: data.id,
      title: data.title,
      content: data.content,
      posX: parseFloat(data.posX) || 50,
      posY: parseFloat(data.posY) || 50,
      width: parseFloat(data.width) || 650,
      height: parseFloat(data.height) || 450,
      caseId: data.caseId,
      version: data.version,
      createdBy: createdBy,
    });
  }
  /**
   * Get current content as text (for search/analysis)
   */
  getTextContent(): string {
    const extractText = (nodes: ContentNode[]): string => {
      return nodes
        .map((node) => {
          if (node.text) {
            return node.text;
          }
          if (node.children) {
            return extractText(node.children);
          }
          return "";
        })
        .join("");
    };

    let content = "";
    this.content.subscribe((value) => {
      content = extractText(value);
    })();

    return content;
  }
  /**
   * Get word count
   */
  getWordCount(): number {
    const text = this.getTextContent();
    return text
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0).length;
  }
  /**
   * Clone this report
   */
  clone(): Report {
    const clonedReport = new Report({
      title: "",
      content: [],
      posX: 0,
      posY: 0,
      width: 0,
      height: 0,
      caseId: this.caseId,
      createdBy: this.createdBy,
    });

    // Subscribe to get current values and clone them
    this.title.subscribe((title) =>
      clonedReport.title.set(`${title} (Copy)`),
    )();
    this.content.subscribe((content) =>
      clonedReport.content.set(JSON.parse(JSON.stringify(content))),
    )();
    this.position.subscribe((pos) =>
      clonedReport.position.set({ x: pos.x + 20, y: pos.y + 20 }),
    )();
    this.size.subscribe((size) => clonedReport.size.set({ ...size }))();

    return clonedReport;
  }
}