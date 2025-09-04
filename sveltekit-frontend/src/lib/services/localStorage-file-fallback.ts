/**
 * localStorage File Upload Fallback Service
 * Provides file storage fallback when MinIO/upload services are unavailable
 * Stores file metadata and base64 content in localStorage for offline functionality
 */

import { browser } from '$app/environment';

interface StoredFile {
  id: string;
  name: string;
  type: string;
  size: number;
  content: string; // base64 encoded
  uploadedAt: string;
  caseId?: string;
  description?: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

interface UploadResult {
  success: boolean;
  fileId?: string;
  error?: string;
  url?: string;
  fallback: boolean;
}

const STORAGE_KEY = 'deeds_web_files';
const MAX_STORAGE_SIZE = 50 * 1024 * 1024; // 50MB limit for localStorage

export class LocalStorageFileFallback {
  private static instance: LocalStorageFileFallback;

  static getInstance(): LocalStorageFileFallback {
    if (!LocalStorageFileFallback.instance) {
      LocalStorageFileFallback.instance = new LocalStorageFileFallback();
    }
    return LocalStorageFileFallback.instance;
  }

  /**
   * Check if localStorage is available and has space
   */
  isAvailable(): boolean {
    if (!browser) return false;
    
    try {
      const testKey = '__localStorage_test__';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get current storage usage
   */
  getStorageUsage(): { used: number; available: number; percentage: number } {
    if (!this.isAvailable()) return { used: 0, available: 0, percentage: 0 };

    const files = this.getAllFiles();
    const used = JSON.stringify(files).length;
    const available = MAX_STORAGE_SIZE - used;
    const percentage = (used / MAX_STORAGE_SIZE) * 100;

    return { used, available, percentage };
  }

  /**
   * Upload file to localStorage
   */
  async uploadFile(
    file: File, 
    metadata: { caseId?: string; description?: string; tags?: string[] } = {}
  ): Promise<UploadResult> {
    if (!this.isAvailable()) {
      return { success: false, error: 'localStorage not available', fallback: true };
    }

    try {
      // Check file size
      if (file.size > 10 * 1024 * 1024) { // 10MB per file limit
        return { success: false, error: 'File too large (max 10MB)', fallback: true };
      }

      // Check available storage
      const usage = this.getStorageUsage();
      const estimatedFileSize = file.size * 1.4; // Base64 overhead
      if (usage.available < estimatedFileSize) {
        return { success: false, error: 'Insufficient storage space', fallback: true };
      }

      // Convert file to base64
      const content = await this.fileToBase64(file);
      
      const storedFile: StoredFile = {
        id: this.generateId(),
        name: file.name,
        type: file.type,
        size: file.size,
        content,
        uploadedAt: new Date().toISOString(),
        ...metadata
      };

      // Save to localStorage
      const files = this.getAllFiles();
      files.push(storedFile);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(files));

      console.log('📦 File stored in localStorage fallback:', storedFile.name);

      return {
        success: true,
        fileId: storedFile.id,
        url: `localStorage://${storedFile.id}`,
        fallback: true
      };

    } catch (error: any) {
      console.error('❌ localStorage file upload failed:', error);
      return { success: false, error: error.message, fallback: true };
    }
  }

  /**
   * Get file from localStorage
   */
  getFile(fileId: string): StoredFile | null {
    if (!this.isAvailable()) return null;

    const files = this.getAllFiles();
    return files.find(f => f.id === fileId) || null;
  }

  /**
   * Get all files from localStorage
   */
  getAllFiles(): StoredFile[] {
    if (!this.isAvailable()) return [];

    try {
      const data = localStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch {
      return [];
    }
  }

  /**
   * Get files for a specific case
   */
  getFilesByCase(caseId: string): StoredFile[] {
    return this.getAllFiles().filter(f => f.caseId === caseId);
  }

  /**
   * Delete file from localStorage
   */
  deleteFile(fileId: string): boolean {
    if (!this.isAvailable()) return false;

    try {
      const files = this.getAllFiles();
      const filteredFiles = files.filter(f => f.id !== fileId);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filteredFiles));
      console.log('🗑️ File deleted from localStorage:', fileId);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create downloadable URL for localStorage file
   */
  createDownloadUrl(fileId: string): string | null {
    const file = this.getFile(fileId);
    if (!file) return null;

    try {
      // Convert base64 back to blob
      const byteCharacters = atob(file.content);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: file.type })
      
      return URL.createObjectURL(blob);
    } catch {
      return null;
    }
  }

  /**
   * Clear all files (cleanup)
   */
  clearAllFiles(): boolean {
    if (!this.isAvailable()) return false;

    try {
      localStorage.removeItem(STORAGE_KEY);
      console.log('🧹 localStorage files cleared');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get file statistics
   */
  getStats(): {
    totalFiles: number;
    totalSize: number;
    fileTypes: Record<string, number>;
    casesWithFiles: string[];
  } {
    const files = this.getAllFiles();
    
    return {
      totalFiles: files.length,
      totalSize: files.reduce((sum, f) => sum + f.size, 0),
      fileTypes: files.reduce((types, f) => {
        const ext = f.name.split('.').pop()?.toLowerCase() || 'unknown';
        types[ext] = (types[ext] || 0) + 1;
        return types;
      }, {} as Record<string, number>),
      casesWithFiles: [...new Set(files.filter(f => f.caseId).map(f => f.caseId!))]
    };
  }

  // Helper methods

  private fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64 = (reader.result as string).split(',')[1];
        resolve(base64);
      };
      reader.onerror = error => reject(error);
    });
  }

  private generateId(): string {
    return `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Export singleton instance
export const localStorageFiles = LocalStorageFileFallback.getInstance();