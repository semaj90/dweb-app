// Minimal ambient types for service worker events used in this repo
// Keeps the incremental TypeScript cleanup low-risk by declaring commonly used members

declare interface ExtendableEvent extends Event {
  waitUntil(promise: Promise<unknown>): void;
}

declare interface FetchEvent extends ExtendableEvent {
  request: Request;
  respondWith(response: Response | Promise<Response>): void;
}

declare interface ExtendableMessageEvent extends ExtendableEvent, MessageEvent {
}

declare global {
  interface ServiceWorkerGlobalScope {
    skipWaiting(): Promise<void> | void;
    clients: any;
  }

  const self: ServiceWorkerGlobalScope & typeof globalThis;
}

export {};
