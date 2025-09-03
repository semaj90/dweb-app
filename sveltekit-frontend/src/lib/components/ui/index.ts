
// Comprehensive UI Component Exports - Multi-Library Support
// Support for both bits-ui and melt-ui v0.39.0 (Svelte 5 Compatible)
// Auto-generated barrel file for all UI components

// Deprecated Melt UI exports removed; consolidate on Bits UI + enhanced variants

// === BITS UI V2 COMPONENTS (SVELTE 5 COMPATIBLE) ===
// Primary exports - Production ready components
export * from "./dialog";
export * from "./command";

// === MODULAR API-INTEGRATED COMPONENTS ===
export { default as ModularDialog } from './modular-dialog/ModularDialog.svelte';
export { default as ModularCommand } from './modular-command/ModularCommand.svelte';
export { default as CaseManagementExample } from './modular-examples/CaseManagementExample.svelte';

// Enhanced Bits UI Components (Legal AI specific) - Primary exports
export { default as EnhancedButton } from "./enhanced-bits/Button.svelte";
export { default as EnhancedDialog } from "./enhanced-bits/Dialog.svelte";
export { default as EnhancedInput } from "./enhanced-bits/Input.svelte";

// Standard Component Barrel Exports (no conflicts)
export * from "./label";
export * from "./textarea";
export * from "./badge";
export * from "./tabs";
export * from "./tooltip";
export * from "./progress";
export * from "./scrollarea";
export * from "./layout";
export * from "./modal";

// Individual Svelte component imports for direct use
import Badge from "./Badge.svelte";
import Button from "./Button.svelte";
import Card from "./Card.svelte";
import CardContent from "./CardContent.svelte";
import CardFooter from "./CardFooter.svelte";
import CardHeader from "./CardHeader.svelte";
import CardTitle from "./CardTitle.svelte";
import Input from "./Input.svelte";
import Label from "./Label.svelte";
import Modal from "./Modal.svelte";
import Tooltip from "./Tooltip.svelte";

// Export individual components for direct access
export {
  Badge as UiBadge,
  Button as UiButton,
  Card as UiCard,
  CardContent as UiCardContent,
  CardFooter as UiCardFooter,
  CardHeader as UiCardHeader,
  CardTitle as UiCardTitle,
  Input as UiInput,
  Label as UiLabel,
  Modal as UiModal,
  Tooltip as UiTooltip,
};

// Standard Bits UI v2 components (optimized)
export { default as ContextMenuStandard } from './context-menu/ContextMenuStandard.svelte';
export { default as DialogStandard } from './dialog/DialogStandard.svelte';
export { default as SelectStandard } from './select/SelectStandard.svelte';
export { default as FormStandard } from './forms/FormStandard.svelte';

// Lazy loading for performance optimization
export const LazyCommandMenu = () => import('./CommandMenu.svelte');
export const LazyRichTextEditor = () => import('./RichTextEditor.svelte');
export const LazyMarkdownRenderer = () => import('./MarkdownRenderer.svelte');
export const LazyDragDropZone = () => import('./DragDropZone.svelte');

// Legacy exports for compatibility
export { default as BitsUnoDemo } from "./BitsUnoDemo.svelte";
export { default as CaseForm } from "./CaseForm.svelte";
export { default as CommandMenu } from "./CommandMenu.svelte";
export { default as DragDropZone } from "./DragDropZone.svelte";
export { default as Form } from "./Form.svelte";
export { default as MarkdownRenderer } from "./MarkdownRenderer.svelte";
export { default as RichTextEditor } from "./RichTextEditor.svelte";
export { default as SmartTextarea } from "./SmartTextarea.svelte";

// Performance utilities
export const preloadComponent = async (loader: () => Promise<any>) => {
  try {
    return await loader();
  } catch (error: any) {
    console.warn('Failed to preload component:', error);
    return null;
  }
};

// === COMPONENT LIBRARY CHOICE SYSTEM ===
export type ComponentLibrary = 'bits-ui';

export interface ComponentChoice {
  library: ComponentLibrary;
  reason?: string;
}

// Helper function to choose component library
export function selectComponentLibrary(choice: ComponentLibrary = 'bits-ui'): ComponentChoice {
  return { library: choice, reason: 'Using bits-ui consolidated UI system' };
}

// Re-export bits-ui for direct access
export * as BitsUI from 'bits-ui';

// Default component mappings (Bits UI / internal variants)
export { default as Button } from './bitsbutton.svelte';
