// Shared OCR types to be used across frontend services
export interface OCRResult {
  text: string;
  confidence: number;
  processing_time?: number;
  pages?: Array<{ page: number; text: string }>;
  metadata?: Record<string, any>;
}

export interface OCRWord {
  text: string;
  confidence?: number;
  boundingBox?: { x: number; y: number; width: number; height: number };
}

export default OCRResult;
