/**
 * NES Memory Architecture - Phase 14
 * 
 * Nintendo Entertainment System inspired memory management for legal AI applications
 * Features:
 * - NES-authentic memory regions (2KB RAM, 8KB CHR-ROM, 32KB PRG-ROM)
 * - Legal document context-aware allocation
 * - Memory bank switching for large documents
 * - Performance optimization with 8-bit addressing patterns
 * - Real-time garbage collection with legal priority scoring
 */

// Nintendo NES Memory Map (authentic constraints)
const NES_MEMORY_MAP = {
  // Internal RAM (2KB, mirrored to fill 8KB space)
  INTERNAL_RAM: {
    start: 0x0000,
    end: 0x07FF,
    size: 2048,
    mirrored: true,
    mirrorSize: 8192 // $0000-$1FFF
  },
  
  // PPU registers (for UI components)
  PPU_REGISTERS: {
    start: 0x2000,
    end: 0x2007,
    size: 8,
    mirrored: true,
    mirrorSize: 8192 // $2000-$3FFF
  },
  
  // APU and I/O registers (for audio/input)
  APU_IO_REGISTERS: {
    start: 0x4000,
    end: 0x4017,
    size: 24
  },
  
  // Expansion ROM (for legal plugins)
  EXPANSION_ROM: {
    start: 0x4020,
    end: 0x5FFF,
    size: 8160
  },
  
  // Save RAM (for persistent legal data)
  SAVE_RAM: {
    start: 0x6000,
    end: 0x7FFF,
    size: 8192
  },
  
  // PRG-ROM (Program ROM - for legal processing logic)
  PRG_ROM: {
    start: 0x8000,
    end: 0xFFFF,
    size: 32768,
    bankSwitchable: true
  },
  
  // CHR-ROM (Character ROM - for legal document patterns)
  CHR_ROM: {
    start: 0x0000, // Separate PPU address space
    end: 0x1FFF,
    size: 8192,
    bankSwitchable: true
  }
} as const;

interface LegalDocument {
  id: string;
  type: 'contract' | 'evidence' | 'brief' | 'citation' | 'precedent';
  priority: number; // 0-255 (8-bit)
  size: number;
  confidenceLevel: number; // 0.0-1.0
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  lastAccessed: number;
  bankId?: number; // For large documents requiring bank switching
  compressed: boolean;
  metadata: {
    caseId?: string;
    jurisdiction?: string;
    documentClass?: string;
    aiGenerated?: boolean;
    vectorEmbedding?: Float32Array;
  };
}

interface MemoryBank {
  id: number;
  type: 'RAM' | 'CHR_ROM' | 'PRG_ROM' | 'SAVE_RAM' | 'EXPANSION_ROM';
  startAddress: number;
  endAddress: number;
  size: number;
  used: number;
  documents: Map<string, LegalDocument>;
  isActive: boolean;
  lastBankSwitch: number;
  compressionRatio: number;
}

interface MemoryStats {
  totalRAM: number;
  usedRAM: number;
  totalCHR: number;
  usedCHR: number;
  totalPRG: number;
  usedPRG: number;
  bankSwitches: number;
  garbageCollections: number;
  compressionSavings: number;
  documentCount: number;
  averageAccessTime: number;
}

export class NESMemoryArchitecture {
  private memoryBanks: Map<string, MemoryBank> = new Map();
  private activeBank: string = 'INTERNAL_RAM';
  private bankSwitchCount = 0;
  private gcCount = 0;
  private compressionWorker: Worker | null = null;
  
  // NES-style memory management state
  private memoryState = {
    currentScanline: 0,
    vblankActive: false,
    ppu2000: 0, // PPU control register
    ppu2001: 0, // PPU mask register
    ppu2002: 0, // PPU status register
    oamaddr: 0, // OAM address register
    ppuscroll: { x: 0, y: 0 }, // PPU scroll registers
    ppuaddr: 0, // PPU address register
    ppudata: 0  // PPU data register
  };

  // Legal AI priority scoring
  private readonly LEGAL_PRIORITIES = {
    critical: 255,    // Active court case documents
    high: 192,        // Evidence and contracts under review
    medium: 128,      // Research documents and precedents
    low: 64,          // Archived or reference materials
    background: 32    // AI-generated summaries and metadata
  };

  constructor() {
    this.initializeMemoryBanks();
    this.setupCompressionWorker();
    this.startVBlankCycle();
  }

  private initializeMemoryBanks() {
    // Internal RAM bank (2KB - most frequently accessed)
    this.memoryBanks.set('INTERNAL_RAM', {
      id: 0,
      type: 'RAM',
      startAddress: NES_MEMORY_MAP.INTERNAL_RAM.start,
      endAddress: NES_MEMORY_MAP.INTERNAL_RAM.end,
      size: NES_MEMORY_MAP.INTERNAL_RAM.size,
      used: 0,
      documents: new Map(),
      isActive: true,
      lastBankSwitch: Date.now(),
      compressionRatio: 1.0
    });

    // CHR-ROM bank (8KB - legal document patterns)
    this.memoryBanks.set('CHR_ROM', {
      id: 1,
      type: 'CHR_ROM',
      startAddress: NES_MEMORY_MAP.CHR_ROM.start,
      endAddress: NES_MEMORY_MAP.CHR_ROM.end,
      size: NES_MEMORY_MAP.CHR_ROM.size,
      used: 0,
      documents: new Map(),
      isActive: true,
      lastBankSwitch: Date.now(),
      compressionRatio: 1.0
    });

    // PRG-ROM bank (32KB - legal processing logic)
    this.memoryBanks.set('PRG_ROM', {
      id: 2,
      type: 'PRG_ROM',
      startAddress: NES_MEMORY_MAP.PRG_ROM.start,
      endAddress: NES_MEMORY_MAP.PRG_ROM.end,
      size: NES_MEMORY_MAP.PRG_ROM.size,
      used: 0,
      documents: new Map(),
      isActive: true,
      lastBankSwitch: Date.now(),
      compressionRatio: 1.0
    });

    // Save RAM bank (8KB - persistent legal data)
    this.memoryBanks.set('SAVE_RAM', {
      id: 3,
      type: 'SAVE_RAM',
      startAddress: NES_MEMORY_MAP.SAVE_RAM.start,
      endAddress: NES_MEMORY_MAP.SAVE_RAM.end,
      size: NES_MEMORY_MAP.SAVE_RAM.size,
      used: 0,
      documents: new Map(),
      isActive: true,
      lastBankSwitch: Date.now(),
      compressionRatio: 1.0
    });

    // Expansion ROM bank (8KB - legal plugins)
    this.memoryBanks.set('EXPANSION_ROM', {
      id: 4,
      type: 'EXPANSION_ROM',
      startAddress: NES_MEMORY_MAP.EXPANSION_ROM.start,
      endAddress: NES_MEMORY_MAP.EXPANSION_ROM.end,
      size: NES_MEMORY_MAP.EXPANSION_ROM.size,
      used: 0,
      documents: new Map(),
      isActive: false, // Activated on-demand
      lastBankSwitch: Date.now(),
      compressionRatio: 1.0
    });
  }

  private setupCompressionWorker() {
    const workerCode = `
      // Legal document compression with NES-style RLE encoding
      self.onmessage = function(e) {
        const { documentData, legalContext, compressionLevel } = e.data;
        
        try {
          const compressed = compressLegalDocument(documentData, legalContext, compressionLevel);
          
          self.postMessage({
            success: true,
            compressedData: compressed.data,
            compressionRatio: compressed.ratio,
            originalSize: documentData.byteLength,
            compressedSize: compressed.data.byteLength,
            legalPriority: compressed.legalPriority
          });
        } catch (error) {
          self.postMessage({ success: false, error: error.message });
        }
      };
      
      function compressLegalDocument(data, legalContext, level) {
        const input = new Uint8Array(data);
        const output = [];
        let i = 0;
        
        // Legal context-aware compression
        const priorityMultiplier = legalContext?.riskLevel === 'critical' ? 3 : 
                                 legalContext?.riskLevel === 'high' ? 2 : 1;
        const confidenceBoost = legalContext?.confidenceLevel > 0.8 ? 1.5 : 1.0;
        
        while (i < input.length) {
          const value = input[i];
          let count = 1;
          
          // NES-style RLE with legal document patterns
          const maxRun = Math.min(255, 127 * priorityMultiplier * confidenceBoost);
          while (i + count < input.length && input[i + count] === value && count < maxRun) {
            count++;
          }
          
          if (count > 3 || (legalContext?.type === 'contract' && count > 2)) {
            // Use RLE for sequences (legal documents often have repetitive structure)
            output.push(0xFF, count, value);
          } else {
            // Raw data for short sequences
            for (let j = 0; j < count; j++) {
              output.push(value);
            }
          }
          
          i += count;
        }
        
        const compressed = new Uint8Array(output);
        const ratio = input.length / compressed.length;
        
        // Calculate legal priority based on compression characteristics
        let legalPriority = 128; // default medium priority
        if (ratio > 3.0) legalPriority = 192; // highly structured document
        if (ratio < 1.5) legalPriority = 64;  // complex/unique document
        if (legalContext?.riskLevel === 'critical') legalPriority = 255;
        
        return {
          data: compressed.buffer,
          ratio: ratio,
          legalPriority: legalPriority
        };
      }
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    this.compressionWorker = new Worker(URL.createObjectURL(blob));
  }

  private startVBlankCycle() {
    // NES-style VBlank cycle for memory management (60Hz)
    setInterval(() => {
      this.memoryState.vblankActive = true;
      this.performVBlankOperations();
      
      setTimeout(() => {
        this.memoryState.vblankActive = false;
      }, 1350); // VBlank period (~1.35ms)
      
      this.memoryState.currentScanline = (this.memoryState.currentScanline + 1) % 262;
    }, 16.67); // ~60 FPS
  }

  private performVBlankOperations() {
    // Use VBlank period for memory management operations
    if (this.memoryState.currentScanline % 60 === 0) {
      // Every second, check for garbage collection
      this.checkGarbageCollection();
    }
    
    if (this.memoryState.currentScanline % 180 === 0) {
      // Every 3 seconds, optimize memory layout
      this.optimizeMemoryLayout();
    }
  }

  async allocateDocument(
    document: Omit<LegalDocument, 'lastAccessed'>,
    data: ArrayBuffer,
    options: {
      preferredBank?: string;
      compress?: boolean;
      compressionLevel?: number;
    } = {}
  ): Promise<boolean> {
    const {
      preferredBank = this.selectOptimalBank(document, data.byteLength),
      compress = true,
      compressionLevel = 2
    } = options;

    try {
      const bank = this.memoryBanks.get(preferredBank);
      if (!bank) {
        throw new Error(`Invalid memory bank: ${preferredBank}`);
      }

      // Check if document fits in bank
      let documentSize = data.byteLength;
      let finalData = data;
      let compressionRatio = 1.0;

      if (compress && this.compressionWorker) {
        const compressed = await this.compressDocument(data, document, compressionLevel);
        finalData = compressed.data;
        documentSize = compressed.data.byteLength;
        compressionRatio = compressed.ratio;
      }

      if (bank.used + documentSize > bank.size) {
        // Try garbage collection
        await this.garbageCollectBank(preferredBank);
        
        if (bank.used + documentSize > bank.size) {
          // Try bank switching if supported
          if ((bank.type === 'PRG_ROM' || bank.type === 'CHR_ROM') && 
               NES_MEMORY_MAP[bank.type].bankSwitchable) {
            const success = await this.performBankSwitch(preferredBank, document);
            if (!success) {
              console.warn(`❌ Cannot allocate document ${document.id}: insufficient memory`);
              return false;
            }
          } else {
            return false;
          }
        }
      }

      // Calculate legal priority
      const priority = this.calculateLegalPriority(document);

      const legalDocument: LegalDocument = {
        ...document,
        priority,
        size: documentSize,
        lastAccessed: Date.now(),
        compressed: compress,
        bankId: bank.id
      };

      // Allocate document in bank
      bank.documents.set(document.id, legalDocument);
      bank.used += documentSize;
      bank.compressionRatio = bank.compressionRatio * 0.9 + compressionRatio * 0.1;

      console.log(`✅ Allocated ${document.type} document ${document.id} in ${preferredBank} (${this.formatBytes(documentSize)})`);
      return true;

    } catch (error) {
      console.error(`❌ Failed to allocate document ${document.id}:`, error);
      return false;
    }
  }

  private selectOptimalBank(document: Omit<LegalDocument, 'lastAccessed'>, size: number): string {
    // NES-style bank selection based on document characteristics
    
    // Critical legal documents go to fast RAM
    if (document.riskLevel === 'critical' || document.confidenceLevel > 0.9) {
      if (this.memoryBanks.get('INTERNAL_RAM')!.used + size <= this.memoryBanks.get('INTERNAL_RAM')!.size) {
        return 'INTERNAL_RAM';
      }
    }

    // Contracts and evidence patterns go to CHR-ROM
    if (document.type === 'contract' || document.type === 'evidence') {
      if (this.memoryBanks.get('CHR_ROM')!.used + size <= this.memoryBanks.get('CHR_ROM')!.size) {
        return 'CHR_ROM';
      }
    }

    // Legal processing logic goes to PRG-ROM
    if (document.type === 'brief' || document.type === 'precedent') {
      if (this.memoryBanks.get('PRG_ROM')!.used + size <= this.memoryBanks.get('PRG_ROM')!.size) {
        return 'PRG_ROM';
      }
    }

    // Persistent data goes to Save RAM
    if (document.metadata?.caseId) {
      if (this.memoryBanks.get('SAVE_RAM')!.used + size <= this.memoryBanks.get('SAVE_RAM')!.size) {
        return 'SAVE_RAM';
      }
    }

    // Default to PRG-ROM (largest bank)
    return 'PRG_ROM';
  }

  private calculateLegalPriority(document: Omit<LegalDocument, 'lastAccessed' | 'priority'>): number {
    let priority = this.LEGAL_PRIORITIES.medium; // Base priority

    // Risk level adjustment
    switch (document.riskLevel) {
      case 'critical':
        priority = this.LEGAL_PRIORITIES.critical;
        break;
      case 'high':
        priority = this.LEGAL_PRIORITIES.high;
        break;
      case 'medium':
        priority = this.LEGAL_PRIORITIES.medium;
        break;
      case 'low':
        priority = this.LEGAL_PRIORITIES.low;
        break;
    }

    // Confidence level adjustment (0-31 range)
    const confidenceBoost = Math.floor(document.confidenceLevel * 31);
    priority = Math.min(255, priority + confidenceBoost);

    // Document type adjustment
    if (document.type === 'evidence') priority += 16;
    if (document.type === 'contract') priority += 8;
    if (document.metadata?.aiGenerated) priority -= 16;

    return Math.max(0, Math.min(255, priority));
  }

  private async compressDocument(
    data: ArrayBuffer,
    document: Omit<LegalDocument, 'lastAccessed'>,
    compressionLevel: number
  ): Promise<{ data: ArrayBuffer; ratio: number; priority: number }> {
    if (!this.compressionWorker) {
      return { data, ratio: 1.0, priority: 128 };
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Compression timeout'));
      }, 5000);

      this.compressionWorker!.onmessage = (e) => {
        clearTimeout(timeout);
        if (e.data.success) {
          resolve({
            data: e.data.compressedData,
            ratio: e.data.compressionRatio,
            priority: e.data.legalPriority
          });
        } else {
          reject(new Error(e.data.error));
        }
      };

      this.compressionWorker!.postMessage({
        documentData: data,
        legalContext: {
          type: document.type,
          riskLevel: document.riskLevel,
          confidenceLevel: document.confidenceLevel
        },
        compressionLevel
      });
    });
  }

  private async performBankSwitch(bankName: string, document: Omit<LegalDocument, 'lastAccessed'>): Promise<boolean> {
    const bank = this.memoryBanks.get(bankName);
    if (!bank || !NES_MEMORY_MAP[bank.type].bankSwitchable) {
      return false;
    }

    try {
      // Find least important documents to swap out
      const documents = Array.from(bank.documents.entries());
      documents.sort((a, b) => {
        const [, docA] = a;
        const [, docB] = b;
        
        // Sort by priority (low first) and last access time (old first)
        if (docA.priority !== docB.priority) {
          return docA.priority - docB.priority;
        }
        return docA.lastAccessed - docB.lastAccessed;
      });

      // Swap out documents until we have enough space
      const requiredSpace = document.size || 1024; // Estimate if not provided
      let freedSpace = 0;
      const swappedDocs: string[] = [];

      for (const [docId, doc] of documents) {
        if (freedSpace >= requiredSpace) break;
        
        // Don't swap critical documents
        if (doc.riskLevel === 'high' || doc.priority > 200) continue;

        // Move to expansion ROM or remove if not critical
        if (doc.riskLevel === 'low' || doc.riskLevel === 'medium') {
          await this.swapToExpansionROM(docId, doc);
          bank.documents.delete(docId);
          bank.used -= doc.size;
          freedSpace += doc.size;
          swappedDocs.push(docId);
        }
      }

      this.bankSwitchCount++;
      bank.lastBankSwitch = Date.now();

      console.log(`🔄 Bank switch in ${bankName}: swapped ${swappedDocs.length} documents, freed ${this.formatBytes(freedSpace)}`);
      return freedSpace >= requiredSpace;

    } catch (error) {
      console.error(`❌ Bank switch failed in ${bankName}:`, error);
      return false;
    }
  }

  private async swapToExpansionROM(docId: string, document: LegalDocument): Promise<void> {
    const expansionBank = this.memoryBanks.get('EXPANSION_ROM');
    if (!expansionBank) return;

    if (expansionBank.used + document.size <= expansionBank.size) {
      expansionBank.documents.set(docId, {
        ...document,
        bankId: expansionBank.id
      });
      expansionBank.used += document.size;
      expansionBank.isActive = true;
    }
  }

  private async garbageCollectBank(bankName: string): Promise<void> {
    const bank = this.memoryBanks.get(bankName);
    if (!bank) return;

    const documents = Array.from(bank.documents.entries());
    const currentTime = Date.now();
    const oldThreshold = 5 * 60 * 1000; // 5 minutes
    
    let freedSpace = 0;
    const removedDocs: string[] = [];

    for (const [docId, document] of documents) {
      const age = currentTime - document.lastAccessed;
      
      // Remove old, low-priority documents
      if (age > oldThreshold && document.priority < 100 && document.riskLevel !== 'critical') {
        bank.documents.delete(docId);
        bank.used -= document.size;
        freedSpace += document.size;
        removedDocs.push(docId);
      }
    }

    if (removedDocs.length > 0) {
      this.gcCount++;
      console.log(`🗑️ Garbage collected ${removedDocs.length} documents from ${bankName}, freed ${this.formatBytes(freedSpace)}`);
    }
  }

  private checkGarbageCollection(): void {
    for (const [bankName, bank] of this.memoryBanks) {
      const utilizationRate = bank.used / bank.size;
      
      if (utilizationRate > 0.85) {
        this.garbageCollectBank(bankName);
      }
    }
  }

  private optimizeMemoryLayout(): void {
    // NES-style memory optimization during VBlank
    for (const [bankName, bank] of this.memoryBanks) {
      if (bank.documents.size === 0) continue;

      const documents = Array.from(bank.documents.values());
      documents.sort((a, b) => {
        // Sort by access frequency and priority
        const accessWeight = (Date.now() - a.lastAccessed) / 1000;
        const priorityWeight = (255 - a.priority) / 255;
        return (accessWeight + priorityWeight) - (Date.now() - b.lastAccessed) / 1000 + (255 - b.priority) / 255;
      });

      // Update internal layout for better cache locality
      // (This would typically involve actual memory reorganization)
    }
  }

  getDocument(documentId: string): LegalDocument | null {
    for (const bank of this.memoryBanks.values()) {
      const document = bank.documents.get(documentId);
      if (document) {
        document.lastAccessed = Date.now();
        return document;
      }
    }
    return null;
  }

  removeDocument(documentId: string): boolean {
    for (const bank of this.memoryBanks.values()) {
      if (bank.documents.has(documentId)) {
        const document = bank.documents.get(documentId)!;
        bank.documents.delete(documentId);
        bank.used -= document.size;
        return true;
      }
    }
    return false;
  }

  getMemoryStats(): MemoryStats {
    let totalRAM = 0, usedRAM = 0;
    let totalCHR = 0, usedCHR = 0;
    let totalPRG = 0, usedPRG = 0;
    let documentCount = 0;
    let totalAccessTime = 0;
    let accessCount = 0;

    for (const bank of this.memoryBanks.values()) {
      documentCount += bank.documents.size;
      
      switch (bank.type) {
        case 'RAM':
          totalRAM += bank.size;
          usedRAM += bank.used;
          break;
        case 'CHR_ROM':
          totalCHR += bank.size;
          usedCHR += bank.used;
          break;
        case 'PRG_ROM':
          totalPRG += bank.size;
          usedPRG += bank.used;
          break;
      }

      for (const doc of bank.documents.values()) {
        const accessTime = Date.now() - doc.lastAccessed;
        totalAccessTime += accessTime;
        accessCount++;
      }
    }

    return {
      totalRAM,
      usedRAM,
      totalCHR,
      usedCHR,
      totalPRG,
      usedPRG,
      bankSwitches: this.bankSwitchCount,
      garbageCollections: this.gcCount,
      compressionSavings: this.calculateCompressionSavings(),
      documentCount,
      averageAccessTime: accessCount > 0 ? totalAccessTime / accessCount : 0
    };
  }

  private calculateCompressionSavings(): number {
    let totalSavings = 0;
    for (const bank of this.memoryBanks.values()) {
      if (bank.compressionRatio > 1.0) {
        const uncompressedSize = bank.used * bank.compressionRatio;
        totalSavings += uncompressedSize - bank.used;
      }
    }
    return totalSavings;
  }

  // PPU-style register access for UI components
  writePPU(register: number, value: number): void {
    switch (register) {
      case 0x2000: // PPU Control
        this.memoryState.ppu2000 = value;
        break;
      case 0x2001: // PPU Mask
        this.memoryState.ppu2001 = value;
        break;
      case 0x2005: // PPU Scroll
        this.memoryState.ppuscroll.x = value;
        break;
      case 0x2006: // PPU Address
        this.memoryState.ppuaddr = value;
        break;
      case 0x2007: // PPU Data
        this.memoryState.ppudata = value;
        break;
    }
  }

  readPPU(register: number): number {
    switch (register) {
      case 0x2002: // PPU Status
        return this.memoryState.ppu2002 | (this.memoryState.vblankActive ? 0x80 : 0);
      case 0x2007: // PPU Data
        return this.memoryState.ppudata;
      default:
        return 0;
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  async destroy(): Promise<void> {
    // Clean up all documents and resources
    for (const bank of this.memoryBanks.values()) {
      bank.documents.clear();
      bank.used = 0;
    }

    if (this.compressionWorker) {
      this.compressionWorker.terminate();
      this.compressionWorker = null;
    }

    this.memoryBanks.clear();
    console.log('🎮 NES Memory Architecture destroyed');
  }
}

// Export singleton instance
export const nesMemory = new NESMemoryArchitecture();

// Export types for use in other modules
export type { LegalDocument, MemoryBank, MemoryStats };