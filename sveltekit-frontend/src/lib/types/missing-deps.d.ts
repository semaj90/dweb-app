// Type declarations for optional dependencies that may not be installed

declare module 'gl-matrix' {
  export const mat4: any;
  export const vec3: any;
  export const quat: any;
}

declare module 'fuse' {
  export default class Fuse<T> {
    constructor(list: T[], options?: any);
    search(query: string): any[];
    _docs?: T[];
  }
}

declare module 'fuse.js' {
  export default class Fuse<T> {
    constructor(list: T[], options?: any);
    search(query: string): any[];
    _docs?: T[];
  }
}

// Tauri types - optional dependency
declare module '@tauri-apps/api/tauri' {
  export function invoke<T = any>(cmd: string, args?: Record<string, any>): Promise<T>;
}

declare module '@tauri-apps/api/fs' {
  export function readTextFile(path: string): Promise<string>;
  export function writeTextFile(path: string, content: string): Promise<void>;
  export function readBinaryFile(path: string): Promise<Uint8Array>;
  export function writeBinaryFile(path: string, content: Uint8Array): Promise<void>;
  export function createDir(path: string, options?: { recursive?: boolean }): Promise<void>;
  export function exists(path: string): Promise<boolean>;
}

// LokiJS types
declare global {
  class Collection<T = any> {
    find(query?: any): T[];
    insert(doc: T): T;
    update(doc: T): T;
    remove(doc: T): void;
  }
}