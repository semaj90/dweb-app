// Minimal ioredis shim to declare commonly-used methods observed in the repo.
// Keep types permissive (any) to reduce noise during the migration.

declare module 'ioredis' {
  export class Redis {
    constructor(...args: any[]);
    get(key: string): Promise<any>;
    set(key: string, value: any): Promise<any>;
    setex?(key: string, seconds: number, value: any): Promise<any>;
    del(key: string): Promise<any>;
    quit(): Promise<any>;
    disconnect?(): void;
    on?(event: string, cb: (...args: any[]) => void): this;
    psubscribe?(pattern: string, cb?: (...args: any[]) => void): Promise<any>;
    publish?(channel: string, message: string): Promise<any>;
    subscribe?(channel: string): Promise<any>;
    psubscribe?(pattern: string): Promise<any>;
    lpush?(key: string, ...values: any[]): Promise<any>;
    rpush?(key: string, ...values: any[]): Promise<any>;
    exists?(key: string): Promise<number>;
    // allow any other access
    [key: string]: any;
  }
}
