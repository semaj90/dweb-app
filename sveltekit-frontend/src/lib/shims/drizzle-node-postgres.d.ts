declare module 'drizzle-orm/node-postgres' {
  export * from 'drizzle-orm/postgres-js';
  const _default: typeof import('drizzle-orm/postgres-js');
  export default _default;
}
