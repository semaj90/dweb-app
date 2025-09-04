import { writable, type Writable } from 'svelte/store';

type HealthLevel = 'unknown' | 'healthy' | 'degraded' | 'unhealthy';

export interface ServiceStatus {
  status: HealthLevel;
  responseTime?: number;
  lastChecked?: Date;
  details?: string;
}

export interface CrossServiceHealth {
  overall: HealthLevel;
  services: Record<string, ServiceStatus>;
  lastChecked: Date;
}

type HealthCheckFn = () => Promise<ServiceStatus>;

/**
 * CrossServiceHealthMonitor
 * - Maintains a Svelte writable store with health for multiple services
 * - Allows registering async health check functions per service
 * - Periodically runs checks and updates the store
 */
export class CrossServiceHealthMonitor {
  private store: Writable<CrossServiceHealth>;
  private checks: Map<string, HealthCheckFn> = new Map();
  private intervalHandle?: ReturnType<typeof setInterval>;
  private running = false;

  constructor(initialServices: string[] = []) {
	const initialServicesRecord: Record<string, ServiceStatus> = {};
	for (const name of initialServices) {
	  initialServicesRecord[name] = { status: 'unknown', lastChecked: undefined };
	}

	this.store = writable<CrossServiceHealth>({
	  overall: 'unknown',
	  services: initialServicesRecord,
	  lastChecked: new Date()
	});
  }

  getStore(): Writable<CrossServiceHealth> {
	return this.store;
  }

  registerService(name: string, checkFn: HealthCheckFn): void {
	if (!name || typeof checkFn !== 'function') return;
	this.checks.set(name, checkFn);
	this.store.update((s) => {
	  if (!s.services[name]) {
		s.services[name] = { status: 'unknown', lastChecked: undefined };
	  }
	  return s;
	});
  }

  unregisterService(name: string): void {
	this.checks.delete(name);
	this.store.update((s) => {
	  delete s.services[name];
	  return s;
	});
  }

  async performChecks(): Promise<any> {
	const entries = Array.from(this.checks.entries());
	const results = await Promise.allSettled(
	  entries.map(([_, fn]) =>
		fn().catch((err) => {
		  return {
			status: 'unhealthy' as HealthLevel,
			details: String(err),
			lastChecked: new Date()
		  } as ServiceStatus;
		})
	  )
	);

	const servicesUpdate: Record<string, ServiceStatus> = {};
	for (let i = 0; i < entries.length; i++) {
	  const name = entries[i][0];
	  const res = results[i];
	  if (res.status === 'fulfilled') {
		servicesUpdate[name] = res.value;
	  } else {
		servicesUpdate[name] = {
		  status: 'unhealthy',
		  details: String((res as PromiseRejectedResult).reason),
		  lastChecked: new Date()
		};
	  }
	}

	// merge with existing services so we don't drop any that have no check registered
	this.store.update((current) => {
	  const merged = { ...current.services };
	  for (const [k, v] of Object.entries(servicesUpdate)) {
		merged[k] = v;
	  }

	  // compute overall health: worst wins
	  let overall: HealthLevel = 'healthy';
	  const levels: HealthLevel[] = Object.values(merged).map((s) => s.status);
	  if (levels.some((l) => l === 'unhealthy')) overall = 'unhealthy';
	  else if (levels.some((l) => l === 'degraded')) overall = 'degraded';
	  else if (levels.length === 0) overall = 'unknown';
	  else overall = 'healthy';

	  return {
		overall,
		services: merged,
		lastChecked: new Date()
	  };
	});
  }

  start(intervalMs = 30000): void {
	if (this.running) return;
	this.running = true;
	// immediate run
	this.performChecks().catch(() => {});
	this.intervalHandle = setInterval(() => {
	  this.performChecks().catch(() => {});
	}, intervalMs);
  }

  stop(): void {
	if (!this.running) return;
	this.running = false;
	if (this.intervalHandle) {
	  clearInterval(this.intervalHandle);
	  this.intervalHandle = undefined;
	}
  }
}

export function createCrossServiceHealthMonitor(initialServices: string[] = []) {
  return new CrossServiceHealthMonitor(initialServices);
}
