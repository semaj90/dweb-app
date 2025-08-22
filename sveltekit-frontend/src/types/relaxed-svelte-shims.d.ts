// Global relaxed shims to reduce noisy TS errors in wide UI layer.
// Narrow types later by replacing usages with concrete interfaces.

// Allow arbitrary data-* / utility / attributify props (UnoCSS etc.)
declare namespace svelteHTML {
  interface HTMLAttributes<T> {
    [attr: `data-${string}`]: unknown;
    [attr: `${string}:${string}`]: unknown; // e.g. border:"~ gray-200"
    mb?: unknown; mt?: unknown; mx?: unknown; my?: unknown; p?: unknown; px?: unknown; py?: unknown; // common utility aliases
    'mb-8'?: unknown; // legacy attributify style key
  }
}

// Broad file preview / import result shims
interface FilePreviewGeneric {
  name: string; size: number; content?: string; raw?: string; data?: unknown; type?: string;
  skipped?: number; imported?: number; updated?: number; errors?: string[];
}
declare global {
  type AnyFilePreview = FilePreviewGeneric;
  interface ImportResultsShape { success: boolean; message: string; results?: { imported: number; updated: number; errors: string[]; skipped?: number }; error?: string }
}

// Local AI model dynamic shape
interface LocalModelDynamic { name: string; status?: string; size?: string; id?: string; isLoaded?: boolean; type?: string; domain?: string; architecture?: string; dimensions?: number; description?: string }

// Event augmentation for custom component events
declare module 'svelte' {
  interface HTMLProps<T> {
    onuploadComplete?: (e: CustomEvent<any>) => void;
    oncommandInsert?: (e: CustomEvent<any>) => void;
  }
}

export {};
