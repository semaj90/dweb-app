// @ts-nocheck
import { db } from "$lib/server/db/index";
import { cases } from "$lib/server/db/unified-schema";
import { redirect } from "@sveltejs/kit";
import { and, count, desc, eq, like, or } from "drizzle-orm";
import type { LayoutServerLoad } from "./$types";

// This runs for /cases and all its sub-pages
// Perfect for loading persistent 3-column layout data
export const load: LayoutServerLoad = async ({ locals, url }) => {
  // Check authentication
  const user = locals.user;
  if (!user) {
    throw redirect(303, "/login");
  }
  // Get search/filter parameters
  const searchQuery = url.searchParams.get("search") || "";
  const statusFilter = url.searchParams.get("status") || "all";
  const priorityFilter = url.searchParams.get("priority") || "all";
  const sortBy = url.searchParams.get("sort") || "openedAt";
  const sortOrder = url.searchParams.get("order") || "desc";

  // Build where conditions
  const whereConditions = [eq(cases.leadProsecutor, user.id)];

  if (searchQuery) {
    whereConditions.push(
      or(
        like(cases.title, `%${searchQuery}%`),
        like(cases.description, `%${searchQuery}%`),
        like(cases.caseNumber, `%${searchQuery}%`)
      )
    );
  }
  if (statusFilter !== "all") {
    whereConditions.push(eq(cases.status, statusFilter));
  }
  if (priorityFilter !== "all") {
    whereConditions.push(eq(cases.priority, priorityFilter));
  }
  // Determine sort order - safely access the cases properties
  let sortColumn;
  try {
    sortColumn = (cases as any)[sortBy] || cases.createdAt;
  } catch {
    sortColumn = cases.createdAt;
  }
  const orderBy = sortOrder === "asc" ? sortColumn : desc(sortColumn);

  // Fetch user's cases with filters applied
  let userCases = [];
  try {
    userCases = await db
      .select({
        id: cases.id,
        title: cases.title,
        caseNumber: cases.caseNumber,
        status: cases.status,
        priority: cases.priority,
        openedAt: cases.createdAt,
        description: cases.description,
        jurisdiction: cases.jurisdiction,
        metadata: cases.metadata,
        prosecutor: cases.leadProsecutor,
      })
      .from(cases)
      .where(and(...whereConditions))
      .orderBy(orderBy)
      .limit(100);
  } catch (error) {
    console.error("Error fetching user cases:", error);
    userCases = [];
  }
  // Get case count for each status (for sidebar stats)
  let caseStats = [];
  try {
    caseStats = await db
      .select({
        status: cases.status,
        count: count(cases.id),
      })
      .from(cases)
      .where(eq(cases.leadProsecutor, user.id))
      .groupBy(cases.status);
  } catch (error) {
    console.error("Error fetching case stats:", error);
    caseStats = [];
  }
  return {
    userCases,
    caseStats,
    searchQuery,
    statusFilter,
    priorityFilter,
    sortBy,
    sortOrder,
    user: user,
  };
};
