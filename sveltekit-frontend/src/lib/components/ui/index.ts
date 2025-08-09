// @ts-nocheck
// Comprehensive UI Component Exports
// Auto-generated barrel file for all UI components

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

// Legacy exports for compatibility
export { default as BitsUnoDemo } from "./BitsUnoDemo.svelte";
export { default as CaseForm } from "./CaseForm.svelte";
export { default as CommandMenu } from "./CommandMenu.svelte";
export { default as DragDropZone } from "./DragDropZone.svelte";
export { default as Form } from "./Form.svelte";
export { default as MarkdownRenderer } from "./MarkdownRenderer.svelte";
export { default as RichTextEditor } from "./RichTextEditor.svelte";
export { default as SmartTextarea } from "./SmartTextarea.svelte";
