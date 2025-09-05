/**
 * Enhanced File Upload Service with localStorage Fallback
 * Attempts upload to primary service (MinIO) and falls back to localStorage
 */

import { browser } from '$app/environment';
import { localStorageFiles } from './localStorage-file-fallback.js';

export interface UploadOptions {
  caseId?: string;
  description?: string;
  tags?: string[];
  useLocalStorage?: boolean; // Force localStorage usage
  maxRetries?: number;
}

export interface UploadResponse {
  success: boolean;
  fileId: string;
  fileName: string;
  url: string;
  size: number;
  error?: string;
  fallbackUsed: boolean;
  storageType: 'server' | 'localStorage';
}

export class EnhancedFileUpload {
  private static instance: EnhancedFileUpload;
  
  static getInstance(): EnhancedFileUpload {
    if (!EnhancedFileUpload.instance) {
      EnhancedFileUpload.instance = new EnhancedFileUpload();
    }
    return EnhancedFileUpload.instance;
  }

  /**
   * Upload file with automatic fallback to localStorage
   */
  async uploadFile(file: File, options: UploadOptions = {}): Promise<UploadResponse> {
    const { caseId, description, tags, useLocalStorage = false, maxRetries = 1 } = options;

    // Force localStorage if requested
    if (useLocalStorage) {
      return this.uploadToLocalStorage(file, { caseId, description, tags });
    }

    // Try server upload first
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const serverResult = await this.uploadToServer(file, { caseId, description, tags });
        if (serverResult.success) {
          return {
            ...serverResult,
            fallbackUsed: false,
            storageType: 'server'
          };
        }
      } catch (error) {
        console.warn(`Server upload attempt ${attempt + 1} failed:`, error);
      }
    }

    // Server upload failed, fallback to localStorage
    console.log('📦 Server upload failed, falling back to localStorage...');
    return this.uploadToLocalStorage(file, { caseId, description, tags });
  }

  /**
   * Upload multiple files with progress tracking
   */
  async uploadFiles(
    files: FileList | File[], 
    options: UploadOptions = {},
    onProgress?: (completed: number, total: number, currentFile: string) => void
  ): Promise<UploadResponse[]> {
    const fileArray = Array.from(files);
    const results: UploadResponse[] = [];

    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      onProgress?.(i, fileArray.length, file.name);

      try {
        const result = await this.uploadFile(file, options);
        results.push(result);
      } catch (error: any) {
        results.push({
          success: false,
          fileId: '',
          fileName: file.name,
          url: '',
          size: file.size,
          error: error.message,
          fallbackUsed: true,
          storageType: 'localStorage'
        });
      }
    }

    onProgress?.(fileArray.length, fileArray.length, 'Complete');
    return results;
  }

  /**
   * Get file URL (handles both server and localStorage)
   */
  getFileUrl(fileId: string, storageType: 'server' | 'localStorage'): string | null {
    if (storageType === 'localStorage') {
      return localStorageFiles.createDownloadUrl(fileId);
    }
    
    // Server file URL
    return `/api/files/${fileId}`;
  }

  /**
   * Delete file (handles both server and localStorage)
   */
  async deleteFile(fileId: string, storageType: 'server' | 'localStorage'): Promise<boolean> {
    if (storageType === 'localStorage') {
      return localStorageFiles.deleteFile(fileId);
    }

    try {
      const response = await fetch(`/api/files/${fileId}`, { method: 'DELETE' });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get storage statistics
   */
  getStorageStats(): {
    localStorage: ReturnType<typeof localStorageFiles.getStats>;
    usage: ReturnType<typeof localStorageFiles.getStorageUsage>;
  } {
    return {
      localStorage: localStorageFiles.getStats(),
      usage: localStorageFiles.getStorageUsage()
    };
  }

  // Private methods

  private async uploadToServer(
    file: File, 
    metadata: { caseId?: string; description?: string; tags?: string[] }
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));

    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server upload failed: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.error || 'Upload failed');
    }

    return {
      success: true,
      fileId: result.fileId,
      fileName: file.name,
      url: result.url,
      size: file.size,
      fallbackUsed: false,
      storageType: 'server' as const
    };
  }

  private async uploadToLocalStorage(
    file: File,
    metadata: { caseId?: string; description?: string; tags?: string[] }
  ): Promise<UploadResponse> {
    const result = await localStorageFiles.uploadFile(file, metadata);

    if (!result.success) {
      throw new Error(result.error || 'localStorage upload failed');
    }

    return {
      success: true,
      fileId: result.fileId!,
      fileName: file.name,
      url: result.url!,
      size: file.size,
      fallbackUsed: true,
      storageType: 'localStorage'
    };
  }
}

// Export singleton
export const enhancedFileUpload = EnhancedFileUpload.getInstance();