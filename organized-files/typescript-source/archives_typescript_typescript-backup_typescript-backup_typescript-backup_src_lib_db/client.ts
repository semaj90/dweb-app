// Minimal DB client type placeholder to satisfy imports during type-check
export type DBClient = {
  query?: (...args: any[]) => Promise<any>;
  run?: (...args: any[]) => Promise<any>;
  close?: () => Promise<any>;
};

export const db: DBClient = {
  // runtime modules will replace/augment this implementation
};

export default db;
