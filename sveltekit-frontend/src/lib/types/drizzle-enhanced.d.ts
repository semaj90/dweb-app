/**
 * Drizzle ORM Enhanced Type Definitions
 * Fixes "Untyped function calls may not accept type arguments" errors
 */

declare module 'drizzle-orm/pg-core' {
  // Enhanced table function with proper generics
  export function pgTable<T extends string>(name: T, columns: any, extraConfig?: any): any;
  
  // Enhanced column types with proper generics
  export function serial<T extends string>(name?: T): any;
  export function text<T extends string>(name?: T, config?: any): any;
  export function varchar<T extends string>(name?: T, config?: { length?: number }): any;
  export function integer<T extends string>(name?: T): any;
  export function boolean<T extends string>(name?: T): any;
  export function timestamp<T extends string>(name?: T, config?: any): any;
  export function json<T extends string>(name?: T): any;
  export function jsonb<T extends string>(name?: T): any;
  export function uuid<T extends string>(name?: T): any;
  export function real<T extends string>(name?: T): any;
  export function doublePrecision<T extends string>(name?: T): any;
  export function bigint<T extends string>(name?: T, config?: any): any;
  export function numeric<T extends string>(name?: T, config?: any): any;
  export function decimal<T extends string>(name?: T, config?: any): any;
  export function char<T extends string>(name?: T, config?: any): any;
  export function date<T extends string>(name?: T): any;
  export function time<T extends string>(name?: T, config?: any): any;
  export function interval<T extends string>(name?: T, config?: any): any;
  
  // pgvector specific types
  export function vector<T extends string>(name?: T, config?: { dimensions?: number }): any;
  
  // Constraint functions
  export function primaryKey<T extends any[]>(...columns: T): any;
  export function foreignKey<T extends any>(config: T): any;
  export function unique<T extends any[]>(...columns: T): any;
  export function index<T extends string>(name: T, config?: any): any;
  export function uniqueIndex<T extends string>(name: T, config?: any): any;
  
  // Relations
  export function relations<T extends any>(table: T, relations: any): any;
  export function one<T extends any>(table: T, config?: any): any;
  export function many<T extends any>(table: T, config?: any): any;
}

declare module 'drizzle-orm' {
  // Enhanced SQL operations with proper generics
  export const sql: {
    <T = any>(strings: TemplateStringsArray, ...values: any[]): T;
    raw<T = any>(query: string): T;
    empty(): any;
    fromList<T = any>(list: T[]): T;
  };
  
  // Enhanced query operators
  export function eq<T, U>(left: T, right: U): any;
  export function ne<T, U>(left: T, right: U): any;
  export function gt<T, U>(left: T, right: U): any;
  export function gte<T, U>(left: T, right: U): any;
  export function lt<T, U>(left: T, right: U): any;
  export function lte<T, U>(left: T, right: U): any;
  export function isNull<T>(column: T): any;
  export function isNotNull<T>(column: T): any;
  export function inArray<T>(column: T, values: any[]): any;
  export function notInArray<T>(column: T, values: any[]): any;
  export function like<T>(column: T, pattern: string): any;
  export function notLike<T>(column: T, pattern: string): any;
  export function ilike<T>(column: T, pattern: string): any;
  export function notIlike<T>(column: T, pattern: string): any;
  export function between<T, U>(column: T, min: U, max: U): any;
  export function notBetween<T, U>(column: T, min: U, max: U): any;
  export function exists<T>(query: T): any;
  export function notExists<T>(query: T): any;
  
  // Logical operators
  export function and<T extends any[]>(...conditions: T): any;
  export function or<T extends any[]>(...conditions: T): any;
  export function not<T>(condition: T): any;
  
  // Aggregate functions
  export function count<T>(column?: T): any;
  export function sum<T>(column: T): any;
  export function avg<T>(column: T): any;
  export function min<T>(column: T): any;
  export function max<T>(column: T): any;
  
  // String functions
  export function concat<T extends any[]>(...columns: T): any;
  export function substring<T>(column: T, start: number, length?: number): any;
  export function length<T>(column: T): any;
  export function lower<T>(column: T): any;
  export function upper<T>(column: T): any;
  export function trim<T>(column: T): any;
  
  // Date functions
  export function now(): any;
  export function extract<T>(unit: string, column: T): any;
  
  // Cast functions
  export function cast<T, U>(column: T, type: U): any;
  
  // Case expressions
  export function caseWhen<T>(value?: T): any;
  
  // Window functions
  export function over<T>(fn: T, window?: any): any;
  
  // Array functions (PostgreSQL specific)
  export function arrayContains<T, U>(column: T, value: U): any;
  export function arrayContained<T, U>(column: T, value: U): any;
  export function arrayOverlaps<T, U>(column: T, value: U): any;
  
  // JSON functions (PostgreSQL specific)
  export function jsonExtract<T>(column: T, path: string): any;
  export function jsonArrayLength<T>(column: T): any;
  
  // Vector functions (pgvector specific)
  export function cosineDistance<T, U>(vector1: T, vector2: U): any;
  export function l2Distance<T, U>(vector1: T, vector2: U): any;
  export function innerProduct<T, U>(vector1: T, vector2: U): any;
}

declare module 'drizzle-orm/postgres-js' {
  import type { PostgresJsDatabase } from 'drizzle-orm/postgres-js';
  
  export function drizzle<T = any>(client: any, config?: {
    schema?: T;
    logger?: boolean | any;
  }): PostgresJsDatabase<T>;
}

declare module 'drizzle-orm/node-postgres' {
  export function drizzle<T = any>(client: any, config?: {
    schema?: T;
    logger?: boolean | any;
  }): any;
}

// Enhanced type for better IntelliSense
export type DrizzleTable<T extends Record<string, any> = Record<string, any>> = T;

export interface DrizzleColumn<T = any> {
  dataType: string;
  columnType: string;
  data: T;
  enumValues?: any[];
}

export interface DrizzleQuery<T = any> {
  execute(): Promise<T[]>;
  all(): Promise<T[]>;
  get(): Promise<T | null>;
  values(): Promise<any[][]>;
  raw(): Promise<any>;
}

export interface DrizzleInsert<T = any> {
  values(values: T | T[]): DrizzleQuery<T>;
  returning(): DrizzleQuery<T>;
  returning<U extends keyof T>(columns: U[]): DrizzleQuery<Pick<T, U>>;
  onConflictDoNothing(): DrizzleInsert<T>;
  onConflictDoUpdate(config: any): DrizzleInsert<T>;
}

export interface DrizzleUpdate<T = any> {
  set(values: Partial<T>): DrizzleQuery<T>;
  where(condition: any): DrizzleUpdate<T>;
  returning(): DrizzleQuery<T>;
  returning<U extends keyof T>(columns: U[]): DrizzleQuery<Pick<T, U>>;
}

export interface DrizzleDelete<T = any> {
  where(condition: any): DrizzleQuery<T>;
  returning(): DrizzleQuery<T>;
  returning<U extends keyof T>(columns: U[]): DrizzleQuery<Pick<T, U>>;
}

export interface DrizzleSelect<T = any> {
  from<U>(table: U): DrizzleSelect<T>;
  where(condition: any): DrizzleSelect<T>;
  orderBy(...columns: any[]): DrizzleSelect<T>;
  limit(count: number): DrizzleSelect<T>;
  offset(count: number): DrizzleSelect<T>;
  groupBy(...columns: any[]): DrizzleSelect<T>;
  having(condition: any): DrizzleSelect<T>;
  innerJoin<U>(table: U, condition: any): DrizzleSelect<T>;
  leftJoin<U>(table: U, condition: any): DrizzleSelect<T>;
  rightJoin<U>(table: U, condition: any): DrizzleSelect<T>;
  fullJoin<U>(table: U, condition: any): DrizzleSelect<T>;
  union<U>(query: DrizzleSelect<U>): DrizzleSelect<T | U>;
  unionAll<U>(query: DrizzleSelect<U>): DrizzleSelect<T | U>;
  intersect<U>(query: DrizzleSelect<U>): DrizzleSelect<T>;
  except<U>(query: DrizzleSelect<U>): DrizzleSelect<T>;
  execute(): Promise<T[]>;
  all(): Promise<T[]>;
  get(): Promise<T | null>;
}

export interface DrizzleDatabase<T = any> {
  select(): DrizzleSelect<any>;
  select<U>(columns: U): DrizzleSelect<any>;
  insert<U>(table: U): DrizzleInsert<any>;
  update<U>(table: U): DrizzleUpdate<any>;
  delete<U>(table: U): DrizzleDelete<any>;
  execute(query: any): Promise<any>;
  transaction<U>(callback: (tx: DrizzleDatabase<T>) => Promise<U>): Promise<U>;
}

// Export enhanced types
export type {
  DrizzleTable,
  DrizzleColumn,
  DrizzleQuery,
  DrizzleInsert,
  DrizzleUpdate,
  DrizzleDelete,
  DrizzleSelect,
  DrizzleDatabase
};