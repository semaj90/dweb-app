import { twMerge } from "tailwind-merge";
import { clsx, type ClassValue } from "clsx";

/**
 * Utility function for merging class names using tailwind-merge and clsx
 * This is essential for component prop merging in shadcn-svelte components
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
