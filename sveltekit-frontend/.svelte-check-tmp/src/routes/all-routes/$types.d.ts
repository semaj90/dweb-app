import type * as Kit from '@sveltejs/kit';
import type { RouteDefinition } from '$lib/data/routes-config';

export interface SystemHealthData {
  system_overview: {
    healthy_services: number;
    total_services: number;
    uptime_hours: number;
    last_updated: string;
  };
  services: Array<{
    name: string;
    status: 'healthy' | 'degraded' | 'down';
    port?: number;
    response_time?: number;
  }>;
  performance: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
  };
}

export interface UserSession {
  user: {
    id: string;
    email: string;
    firstName?: string;
    lastName?: string;
    role: 'attorney' | 'paralegal' | 'investigator' | 'user';
    preferences?: {
      theme: string;
      language: string;
      notifications: Record<string, boolean>;
    };
  } | null;
  isAuthenticated: boolean;
}

export interface RecentOperation {
  operation: string;
  timestamp: string;
  status: 'success' | 'error' | 'pending';
  protocol?: string;
}

export interface RoutePageData {
  systemHealth: SystemHealthData | null;
  userSession: UserSession;
  availableRoutes: RouteDefinition[];
  recentOperations: RecentOperation[];
}

type RouteParams = {};
type PageParent = Kit.AwaitedProperties<{}>;

export type PageServerData = RoutePageData;
export type PageData = PageParent & PageServerData;
export type ActionData = unknown;