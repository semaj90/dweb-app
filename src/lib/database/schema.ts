// Compatibility shim - forward to the canonical schema index
// This makes imports from "$lib/database/schema" consistent whether they use the directory or direct file.
export * from './schema/index';
export { default } from './schema/index';
