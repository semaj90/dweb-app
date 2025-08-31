// Compatibility shim - re-export the unified DB surface
// Allows legacy imports like "$lib/database/client" to continue working
export { db, migrationDb, healthCheck, TABLE_NAMES, type Database } from '$lib/server/db';
