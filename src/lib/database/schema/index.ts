// Canonical schema re-exports
// This module provides named and default exports for the various DB schema modules.
// It helps normalize mixed import styles across the codebase.
export * from './legal-documents';

// Default export all schemas
import * as legalDocumentsSchema from './legal-documents';
export default legalDocumentsSchema;
