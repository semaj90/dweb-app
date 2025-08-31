// Permissive Redis (ioredis-like) shim to include commonly used runtime methods.
declare module 'ioredis' {
  class Redis {
    // common methods used in repo
    get(key: string): Promise<any>;
    set(key: string, value: any, ...rest: any[]): Promise<any>;
    setex(key: string, seconds: number, value: any): Promise<any>;
    psubscribe(pattern: string, listener?: (...args: any[]) => void): Promise<any>;
    publish(channel: string, message: string): Promise<any>;
    subscribe(channel: string, listener?: (...args: any[]) => void): Promise<any>;
    disconnect(): void;
    on(event: string, cb: (...args: any[]) => void): void;
    lpush(key: string, ...values: any[]): Promise<any>;
    rpush(key: string, ...values: any[]): Promise<any>;
    exists(key: string): Promise<number>;
    // fallback index signature
    [k: string]: any;
  }

  export = Redis;
  export interface RedisOptions {
    [key: string]: any;
  }
}

// Also provide a generic Redis type for imports that use "Redis" as a type name
declare global {
  type RedisClient = import('ioredis');
}
