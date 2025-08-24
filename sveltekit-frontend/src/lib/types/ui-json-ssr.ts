// UI JSON SSR types for Button component
export interface UIJsonSSRConfig {
  enabled?: boolean;
  renderMode?: 'client' | 'server';
}

export interface ButtonAnalyticsEvent {
  id: string;
  category: string;
  action: string;
  label: string;
  timestamp: number;
  context?: any;
  variant?: string;
  size?: string;
}