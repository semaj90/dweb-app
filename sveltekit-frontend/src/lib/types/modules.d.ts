/**
 * Module declarations for packages without proper TypeScript types
 */

declare module "fuse.js" {
  interface FuseOptions<T> {
    keys?: string[] | { name: string; weight?: number }[];
    threshold?: number;
    distance?: number;
    includeScore?: boolean;
    includeMatches?: boolean;
    minMatchCharLength?: number;
    shouldSort?: boolean;
    findAllMatches?: boolean;
    location?: number;
    ignoreLocation?: boolean;
    ignoreFieldNorm?: boolean;
  }

  interface FuseResult<T> {
    item: T;
    score?: number;
    matches?: Array<{
      indices: Array<[number, number]>;
      value: string;
      key: string;
    }>;
  }

  class Fuse<T> {
    constructor(list: T[], options?: FuseOptions<T>);
    search(query: string): FuseResult<T>[];
    setCollection(list: T[]): void;
    add(item: T): void;
    remove(predicate: (item: T, index: number) => boolean): T[];
  }

  export default Fuse;
}

declare module "lokijs" {
  interface LokiOptions {
    autosave?: boolean;
    autosaveInterval?: number;
    autoload?: boolean;
    autoloadCallback?: (err: any) => void;
    adapter?: any;
    serializationMethod?: string;
    destructureDelimiter?: string;
    persistenceMethod?: string;
    env?: string;
  }

  interface CollectionOptions<T = any> {
    indices?: string | string[];
    unique?: string | string[];
    exact?: string | string[];
    autoupdate?: boolean;
    clone?: boolean;
    cloneMethod?: string;
    asyncListeners?: boolean;
    disableMeta?: boolean;
    disableChangesApi?: boolean;
    disableDeltaChangesApi?: boolean;
    disableFreeze?: boolean;
    ttl?: number;
    ttlInterval?: number;
    transforms?: any;
  }

  interface Collection<T = any> {
    insert(doc: T): T;
    find(query?: any): T[];
    findOne(query?: any): T | null;
    update(doc: T): T;
    remove(doc: T): void;
    removeWhere(query: any): void;
    count(query?: any): number;
    data: T[];
    chain(): any;
    clear(): void;
    where(filter: (obj: T) => boolean): T[];
  }

  export default class Loki {
    constructor(filename?: string, options?: LokiOptions);
    addCollection<T = any>(
      name: string,
      options?: CollectionOptions<T>
    ): Collection<T>;
    getCollection<T = any>(name: string): Collection<T> | null;
    removeCollection(name: string): void;
    saveDatabase(callback?: (err: any) => void): void;
    loadDatabase(options?: any, callback?: (err: any) => void): void;
    close(): void;
  }
}
