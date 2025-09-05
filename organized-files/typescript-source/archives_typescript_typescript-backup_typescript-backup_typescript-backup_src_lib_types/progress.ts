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
      fragment?: any; // partial/streamed result
    }
  | {
      type: 'processing-complete';
      fileId: string;
      finalResult?: any;
    }
  | {
      type: 'error';
      fileId: string;
      error: { message: string; code?: string; meta?: any };
    };