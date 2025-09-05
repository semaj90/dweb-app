import pkg from 'camelcase';

// Some versions of camelcase export the function as module.exports (CJS) and
// some expose a default property when imported from ESM. Normalize both.
const camelcase = pkg && (pkg.default ?? pkg);

export default camelcase;

// Also export a named helper for safety.
export const toCamelCase = (str) => camelcase(str);
