/**
 * Route Registry for Dynamic Route Management
 * Centralized registry for all application routes
 */

import { writable, derived, type Readable } from 'svelte/store';
import { page } from '$app/stores';
import type { RouteDefinition } from '$lib/data/routes-config';
import { allRoutes, getRoutesByCategory } from '$lib/data/routes-config';
import type { GeneratedRoute, DynamicRouteConfig } from './dynamic-route-generator';
import { dynamicRouteGenerator } from './dynamic-route-generator';

export interface RouteRegistryState {
  routes: Map<string, RouteDefinition>;
  dynamicRoutes: Map<string, GeneratedRoute>;
  currentRoute: RouteDefinition | null;
  routeHistory: string[];
  favorites: Set<string>;
  recentRoutes: string[];
}

export interface RouteRegistryOptions {
  maxHistorySize: number;
  maxRecentSize: number;
  persistState: boolean;
  storageKey: string;
}

export class RouteRegistry {
  private state = writable<RouteRegistryState>({
    routes: new Map(),
    dynamicRoutes: new Map(),
    currentRoute: null,
    routeHistory: [],
    favorites: new Set(),
    recentRoutes: []
  });

  private options: RouteRegistryOptions;

  constructor(options: Partial<RouteRegistryOptions> = {}) {
    this.options = {
      maxHistorySize: 50,
      maxRecentSize: 10,
      persistState: true,
      storageKey: 'yorha-route-registry',
      ...options
    };

    this.initialize();
  }

  /**
   * Initialize the registry with existing routes
   */
  private async initialize(): Promise<void> {
    // Load persisted state
    if (this.options.persistState && typeof window !== 'undefined') {
      await this.loadPersistedState();
    }

    // Initialize with static routes
    this.state.update(state => {
      const routesMap = new Map<string, RouteDefinition>();
      for (const route of allRoutes) {
        routesMap.set(route.id, route);
      }
      
      const dynamicRoutesMap = new Map<string, GeneratedRoute>();
      for (const route of dynamicRouteGenerator.getAllRoutes()) {
        dynamicRoutesMap.set(route.id, route);
      }

      return {
        ...state,
        routes: routesMap,
        dynamicRoutes: dynamicRoutesMap
      };
    });

    // Subscribe to page changes for current route tracking
    if (typeof window !== 'undefined') {
      page.subscribe(($page) => {
        this.updateCurrentRoute($page.route.id || $page.url.pathname);
      });
    }

    // Save state on changes
    if (this.options.persistState) {
      this.state.subscribe(() => {
        this.savePersistedState();
      });
    }
  }

  /**
   * Get the current state
   */
  public getState(): Readable<RouteRegistryState> {
    return this.state;
  }

  /**
   * Register a new static route
   */
  public registerRoute(route: RouteDefinition): void {
    this.state.update(state => {
      const newRoutes = new Map(state.routes);
      newRoutes.set(route.id, route);
      
      return {
        ...state,
        routes: newRoutes
      };
    });
  }

  /**
   * Register a new dynamic route
   */
  public registerDynamicRoute(
    id: string,
    path: string,
    config: Partial<DynamicRouteConfig> = {}
  ): GeneratedRoute {
    const generatedRoute = dynamicRouteGenerator.generateRoute(id, path, config);
    
    this.state.update(state => {
      const newDynamicRoutes = new Map(state.dynamicRoutes);
      newDynamicRoutes.set(id, generatedRoute);
      
      return {
        ...state,
        dynamicRoutes: newDynamicRoutes
      };
    });
    
    return generatedRoute;
  }

  /**
   * Unregister a route
   */
  public unregisterRoute(id: string): boolean {
    let removed = false;
    
    this.state.update(state => {
      const newRoutes = new Map(state.routes);
      const newDynamicRoutes = new Map(state.dynamicRoutes);
      
      if (newRoutes.has(id)) {
        newRoutes.delete(id);
        removed = true;
      }
      
      if (newDynamicRoutes.has(id)) {
        newDynamicRoutes.delete(id);
        dynamicRouteGenerator.removeRoute(id);
        removed = true;
      }
      
      return {
        ...state,
        routes: newRoutes,
        dynamicRoutes: newDynamicRoutes
      };
    });
    
    return removed;
  }

  /**
   * Get route by ID
   */
  public getRoute(id: string): RouteDefinition | GeneratedRoute | null {
    const state = this.getCurrentState();
    return state.routes.get(id) || state.dynamicRoutes.get(id) || null;
  }

  /**
   * Get all routes
   */
  public getAllRoutes(): (RouteDefinition | GeneratedRoute)[] {
    const state = this.getCurrentState();
    return [
      ...Array.from(state.routes.values()),
      ...Array.from(state.dynamicRoutes.values())
    ];
  }

  /**
   * Get routes by category
   */
  public getRoutesByCategory(category: string): (RouteDefinition | GeneratedRoute)[] {
    return this.getAllRoutes().filter(route => {
      if ('category' in route) {
        return route.category === category;
      }
      return route.metadata?.category === category;
    });
  }

  /**
   * Search routes
   */
  public searchRoutes(query: string): (RouteDefinition | GeneratedRoute)[] {
    const lowerQuery = query.toLowerCase();
    return this.getAllRoutes().filter(route => {
      const label = 'label' in route ? route.label : route.id;
      const description = 'description' in route ? route.description : '';
      const tags = 'tags' in route ? route.tags : route.metadata?.tags || [];
      
      return (
        label.toLowerCase().includes(lowerQuery) ||
        description.toLowerCase().includes(lowerQuery) ||
        tags.some(tag => tag.toLowerCase().includes(lowerQuery))
      );
    });
  }

  /**
   * Update current route
   */
  private updateCurrentRoute(routeId: string): void {
    this.state.update(state => {
      const route = state.routes.get(routeId) || null;
      const newHistory = [...state.routeHistory];
      const newRecent = [...state.recentRoutes];
      
      // Add to history
      if (routeId && !newHistory.includes(routeId)) {
        newHistory.push(routeId);
        if (newHistory.length > this.options.maxHistorySize) {
          newHistory.shift();
        }
      }
      
      // Add to recent (remove if exists first)
      if (routeId) {
        const existingIndex = newRecent.indexOf(routeId);
        if (existingIndex > -1) {
          newRecent.splice(existingIndex, 1);
        }
        newRecent.unshift(routeId);
        if (newRecent.length > this.options.maxRecentSize) {
          newRecent.pop();
        }
      }
      
      return {
        ...state,
        currentRoute: route,
        routeHistory: newHistory,
        recentRoutes: newRecent
      };
    });
  }

  /**
   * Add route to favorites
   */
  public addToFavorites(routeId: string): void {
    this.state.update(state => ({
      ...state,
      favorites: new Set([...state.favorites, routeId])
    }));
  }

  /**
   * Remove route from favorites
   */
  public removeFromFavorites(routeId: string): void {
    this.state.update(state => {
      const newFavorites = new Set(state.favorites);
      newFavorites.delete(routeId);
      return {
        ...state,
        favorites: newFavorites
      };
    });
  }

  /**
   * Check if route is favorite
   */
  public isFavorite(routeId: string): boolean {
    const state = this.getCurrentState();
    return state.favorites.has(routeId);
  }

  /**
   * Get favorite routes
   */
  public getFavoriteRoutes(): (RouteDefinition | GeneratedRoute)[] {
    const state = this.getCurrentState();
    return Array.from(state.favorites)
      .map(id => this.getRoute(id))
      .filter(route => route !== null) as (RouteDefinition | GeneratedRoute)[];
  }

  /**
   * Get recent routes
   */
  public getRecentRoutes(): (RouteDefinition | GeneratedRoute)[] {
    const state = this.getCurrentState();
    return state.recentRoutes
      .map(id => this.getRoute(id))
      .filter(route => route !== null) as (RouteDefinition | GeneratedRoute)[];
  }

  /**
   * Get route statistics
   */
  public getStatistics(): {
    total: number;
    static: number;
    dynamic: number;
    favorites: number;
    recent: number;
    categories: Record<string, number>;
  } {
    const state = this.getCurrentState();
    const categories: Record<string, number> = {};
    
    for (const route of this.getAllRoutes()) {
      const category = 'category' in route ? route.category : route.metadata?.category || 'unknown';
      categories[category] = (categories[category] || 0) + 1;
    }
    
    return {
      total: state.routes.size + state.dynamicRoutes.size,
      static: state.routes.size,
      dynamic: state.dynamicRoutes.size,
      favorites: state.favorites.size,
      recent: state.recentRoutes.length,
      categories
    };
  }

  /**
   * Clear route history
   */
  public clearHistory(): void {
    this.state.update(state => ({
      ...state,
      routeHistory: [],
      recentRoutes: []
    }));
  }

  /**
   * Clear favorites
   */
  public clearFavorites(): void {
    this.state.update(state => ({
      ...state,
      favorites: new Set()
    }));
  }

  /**
   * Get current state synchronously
   */
  private getCurrentState(): RouteRegistryState {
    let currentState: RouteRegistryState;
    this.state.subscribe(state => {
      currentState = state;
    })();
    return currentState!;
  }

  /**
   * Save state to localStorage
   */
  private savePersistedState(): void {
    if (typeof window === 'undefined') return;
    
    try {
      const state = this.getCurrentState();
      const persistedData = {
        favorites: Array.from(state.favorites),
        recentRoutes: state.recentRoutes,
        routeHistory: state.routeHistory
      };
      
      localStorage.setItem(this.options.storageKey, JSON.stringify(persistedData));
    } catch (error) {
      console.warn('Failed to save route registry state:', error);
    }
  }

  /**
   * Load state from localStorage
   */
  private async loadPersistedState(): Promise<void> {
    if (typeof window === 'undefined') return;
    
    try {
      const saved = localStorage.getItem(this.options.storageKey);
      if (saved) {
        const persistedData = JSON.parse(saved);
        
        this.state.update(state => ({
          ...state,
          favorites: new Set(persistedData.favorites || []),
          recentRoutes: persistedData.recentRoutes || [],
          routeHistory: persistedData.routeHistory || []
        }));
      }
    } catch (error) {
      console.warn('Failed to load route registry state:', error);
    }
  }

  /**
   * Export route manifest
   */
  public exportRouteManifest(): Record<string, any> {
    const state = this.getCurrentState();
    const manifest: Record<string, any> = {};
    
    // Add static routes
    for (const [id, route] of state.routes) {
      manifest[id] = {
        type: 'static',
        ...route
      };
    }
    
    // Add dynamic routes
    for (const [id, route] of state.dynamicRoutes) {
      manifest[id] = {
        type: 'dynamic',
        ...route
      };
    }
    
    return manifest;
  }
}

// Create and export singleton instance
export const routeRegistry = new RouteRegistry();

// Export derived stores for convenient access
export const routes = derived(routeRegistry.getState(), state => 
  Array.from(state.routes.values())
);

export const dynamicRoutes = derived(routeRegistry.getState(), state => 
  Array.from(state.dynamicRoutes.values())
);

export const allRegisteredRoutes = derived(routeRegistry.getState(), state => [
  ...Array.from(state.routes.values()),
  ...Array.from(state.dynamicRoutes.values())
]);

export const currentRoute = derived(routeRegistry.getState(), state => 
  state.currentRoute
);

export const favoriteRoutes = derived(routeRegistry.getState(), state =>
  Array.from(state.favorites)
    .map(id => state.routes.get(id) || state.dynamicRoutes.get(id))
    .filter(Boolean)
);

export const recentRoutes = derived(routeRegistry.getState(), state =>
  state.recentRoutes
    .map(id => state.routes.get(id) || state.dynamicRoutes.get(id))
    .filter(Boolean)
);

export const routeStatistics = derived(routeRegistry.getState(), state => {
  const categories: Record<string, number> = {};
  const allRoutes = [
    ...Array.from(state.routes.values()),
    ...Array.from(state.dynamicRoutes.values())
  ];
  
  for (const route of allRoutes) {
    const category = 'category' in route ? route.category : (route as any).metadata?.category || 'unknown';
    categories[category] = (categories[category] || 0) + 1;
  }
  
  return {
    total: state.routes.size + state.dynamicRoutes.size,
    static: state.routes.size,
    dynamic: state.dynamicRoutes.size,
    favorites: state.favorites.size,
    recent: state.recentRoutes.length,
    categories
  };
});

// Export convenience functions
export function registerRoute(route: RouteDefinition): void {
  return routeRegistry.registerRoute(route);
}

export function registerDynamicRoute(
  id: string,
  path: string,
  config?: Partial<DynamicRouteConfig>
): GeneratedRoute {
  return routeRegistry.registerDynamicRoute(id, path, config);
}

export function unregisterRoute(id: string): boolean {
  return routeRegistry.unregisterRoute(id);
}

export function getRoute(id: string): RouteDefinition | GeneratedRoute | null {
  return routeRegistry.getRoute(id);
}

export function searchRoutes(query: string): (RouteDefinition | GeneratedRoute)[] {
  return routeRegistry.searchRoutes(query);
}

export function addToFavorites(routeId: string): void {
  return routeRegistry.addToFavorites(routeId);
}

export function removeFromFavorites(routeId: string): void {
  return routeRegistry.removeFromFavorites(routeId);
}

export function isFavorite(routeId: string): boolean {
  return routeRegistry.isFavorite(routeId);
}