// Query utilities to handle Drizzle ORM type issues
import type { PgSelectBase } from 'drizzle-orm/pg-core';
import type { SQL } from 'drizzle-orm';
import { and } from 'drizzle-orm';

/**
 * Safely execute a query with proper type handling
 */
export async function executeQuery<T>(query: any): Promise<T[]> {
  return await query;
}

/**
 * Safely execute a count query
 */
export async function executeCountQuery(query: any): Promise<number> {
  const result = await query;
  return result[0]?.count || 0;
}

/**
 * Build a conditional query with proper type handling
 */
export function buildConditionalQuery<T>(
  baseQuery: any,
  conditions: any[]
): any {
  if (conditions.length > 0) {
    return baseQuery.where(conditions.length === 1 ? conditions[0] : and(...conditions));
  }
  return baseQuery;
}

/**
 * Add pagination to a query
 */
export function addPagination<T>(query: any, limit: number, offset: number = 0): any {
  return query.limit(limit).offset(offset);
}