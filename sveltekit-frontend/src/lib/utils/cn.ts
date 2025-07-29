import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Enhanced utility for legal AI applications
export function legalCn(...inputs: ClassValue[]) {
  // Add legal-specific base classes
  const legalBase = "transition-all duration-200 ease-in-out";
  return cn(legalBase, ...inputs);
}

// Confidence-based styling
export function confidenceClass(confidence: number): string {
  if (confidence >= 0.8) {
    return "vector-confidence-high border-green-500 bg-green-50 dark:bg-green-900/20";
  } else if (confidence >= 0.6) {
    return "vector-confidence-medium border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20";
  } else {
    return "vector-confidence-low border-red-500 bg-red-50 dark:bg-red-900/20";
  }
}

// Priority-based styling
export function priorityClass(
  priority: "critical" | "high" | "medium" | "low",
): string {
  const baseClass =
    "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium";

  switch (priority) {
    case "critical":
      return cn(
        baseClass,
        "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400",
      );
    case "high":
      return cn(
        baseClass,
        "bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400",
      );
    case "medium":
      return cn(
        baseClass,
        "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400",
      );
    case "low":
      return cn(
        baseClass,
        "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400",
      );
    default:
      return cn(
        baseClass,
        "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400",
      );
  }
}

// Evidence type styling
export function evidenceTypeClass(type: string): string {
  const baseClass =
    "inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium";

  switch (type) {
    case "document":
      return cn(
        baseClass,
        "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400",
      );
    case "image":
      return cn(
        baseClass,
        "bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400",
      );
    case "video":
      return cn(
        baseClass,
        "bg-pink-100 text-pink-800 dark:bg-pink-900/20 dark:text-pink-400",
      );
    case "audio":
      return cn(
        baseClass,
        "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/20 dark:text-indigo-400",
      );
    case "transcript":
      return cn(
        baseClass,
        "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400",
      );
    default:
      return cn(
        baseClass,
        "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400",
      );
  }
}

// AI status indicators
export function aiStatusClass(
  status: "online" | "processing" | "offline" | "warning",
): string {
  const baseClass = "ai-status-indicator";

  switch (status) {
    case "online":
      return cn(baseClass, "ai-status-online");
    case "processing":
      return cn(baseClass, "ai-status-processing");
    case "offline":
      return cn(baseClass, "ai-status-offline");
    case "warning":
      return cn(baseClass, "ai-status-warning");
    default:
      return cn(baseClass, "ai-status-offline");
  }
}

// NieR theme variants
export function nierVariant(
  variant: "primary" | "secondary" | "accent" | "gothic",
): string {
  switch (variant) {
    case "primary":
      return "bg-nier-bg-primary text-nier-text-primary border-nier-border-primary";
    case "secondary":
      return "bg-nier-bg-secondary text-nier-text-secondary border-nier-border-secondary";
    case "accent":
      return "bg-nier-accent-warm text-nier-text-primary border-nier-border-primary";
    case "gothic":
      return "bg-gothic-bg-primary text-gothic-text-primary border-gothic-border-primary font-gothic";
    default:
      return "bg-nier-bg-primary text-nier-text-primary border-nier-border-primary";
  }
}

// Component state utilities
export function componentState(
  state: "default" | "hover" | "active" | "disabled" | "loading",
): string {
  switch (state) {
    case "hover":
      return "hover:scale-102 hover:shadow-lg transition-transform duration-200";
    case "active":
      return "scale-95 shadow-inner";
    case "disabled":
      return "opacity-50 cursor-not-allowed pointer-events-none";
    case "loading":
      return "animate-pulse cursor-wait";
    default:
      return "";
  }
}

// Responsive utilities for legal workflows
export function responsiveGrid(items: number): string {
  if (items <= 2) return "grid-cols-1 md:grid-cols-2";
  if (items <= 4) return "grid-cols-1 md:grid-cols-2 lg:grid-cols-3";
  if (items <= 6)
    return "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4";
  return "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5";
}

// Animation presets for legal AI
export const animations = {
  slideUp: "animate-slide-up",
  fadeIn: "animate-fade-in",
  processing: "animate-processing",
  pulseSlow: "animate-pulse-slow",
  bounceSubtle: "animate-bounce-subtle",
} as const;

// Legal workflow states
export function workflowState(
  state: "draft" | "review" | "approved" | "rejected" | "archived",
): string {
  const baseClass =
    "inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium";

  switch (state) {
    case "draft":
      return cn(
        baseClass,
        "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400",
      );
    case "review":
      return cn(
        baseClass,
        "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400",
      );
    case "approved":
      return cn(
        baseClass,
        "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400",
      );
    case "rejected":
      return cn(
        baseClass,
        "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400",
      );
    case "archived":
      return cn(
        baseClass,
        "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400",
      );
    default:
      return cn(
        baseClass,
        "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400",
      );
  }
}
