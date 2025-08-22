export type ProgressMsg =
  | {
      type: 'upload-progress';
      fileId: string;
      progress: number; // 0-100
    }
  | {
      type: 'processing-step';
      fileId: string;
      step: 'ocr' | 'embedding' | 'rag' | 'analysis' | string;
      stepProgress?: number; // 0-100
      fragment?: unknown; // partial/streamed result
    }
  | {
      type: 'processing-complete';
      fileId: string;
      finalResult?: unknown;
    }
  | {
      type: 'error';
      fileId: string;
      error: { message: string; code?: string; meta?: unknown };
    };