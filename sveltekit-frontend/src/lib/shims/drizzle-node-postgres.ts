// Shim that re-exports the postgres-js Drizzle adapter so imports
// of 'drizzle-orm/node-postgres' in the codebase resolve to the
// postgres-js adapter implementation used at runtime.
export * from 'drizzle-orm/postgres-js';
import * as _default from 'drizzle-orm/postgres-js';
export default _default;
