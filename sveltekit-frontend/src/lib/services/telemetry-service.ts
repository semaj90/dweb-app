// Lightweight structured telemetry emission service
// Provides in-memory ring buffer + EventTarget based subscription.

export interface TelemetryEvent<T = any> {
  type: string;
  timestamp: string; // ISO
  data: T;
  seq: number;
}

const MAX_EVENTS = 1000;
const buffer: TelemetryEvent[] = [];
let seq = 0;
const target = new EventTarget();

export interface TelemetryEmitOptions {
  consoleDebug?: boolean;
}

function emit<T = any>(type: string, data: T, opts: TelemetryEmitOptions = {}) {
  const evt: TelemetryEvent<T> = { type, data, timestamp: new Date().toISOString(), seq: ++seq };
  buffer.push(evt);
  if (buffer.length > MAX_EVENTS) buffer.splice(0, buffer.length - MAX_EVENTS);
  target.dispatchEvent(new CustomEvent('telemetry', { detail: evt }));
  if (opts.consoleDebug || (globalThis as any).__TELEMETRY_DEBUG__) {
    // eslint-disable-next-line no-console
    console.debug('[telemetry]', evt.type, evt);
  }
  return evt;
}

function subscribe(handler: (event: TelemetryEvent) => void) {
  const listener = (e: Event) => handler((e as CustomEvent).detail);
  target.addEventListener('telemetry', listener as EventListener);
  return () => target.removeEventListener('telemetry', listener as EventListener);
}

function getEvents() { return [...buffer]; }
function clear() { buffer.length = 0; }

export const telemetry = { emit, subscribe, getEvents, clear };
export default telemetry;
