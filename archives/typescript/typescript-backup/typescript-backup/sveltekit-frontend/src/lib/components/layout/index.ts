/**
 * Layout Components Export
 * Layout and structural components for page organization
 */

export { default as MasonryGrid } from './MasonryGrid.svelte';

export type GridLayout = 'masonry' | 'grid' | 'flex' | 'auto';
export type ResponsiveBreakpoint = 'sm' | 'md' | 'lg' | 'xl' | '2xl';

export interface LayoutConfig {
  type: GridLayout;
  columns?: number;
  gap?: number;
  responsive?: Partial<Record<ResponsiveBreakpoint, { columns: number; gap: number }>>;
}