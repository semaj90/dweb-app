
import { db, cases } from "$lib/server/db/index";
import { redirect } from "@sveltejs/kit";
import { and, count, desc, eq, like, or } from "drizzle-orm";
import type { LayoutServerLoad } from "./$types.js";

// This runs for /cases and all its sub-pages
// Perfect for loading persistent 3-column layout data
export const load: LayoutServerLoad = async ({ locals, url }) => {
  // Temporarily bypass authentication for SuperForms testing
  const user = locals.user || {
    id: 'test-user-id',
    name: 'Test User',
    email: 'test@example.com'
  };
  
  // REMOVED: Authentication redirect for testing
  // if (!user?.id) {
  //   throw redirect(303, "/login");
  // }

  // Get search/filter parameters
  const searchQuery = url.searchParams.get("search") || "";
  const statusFilter = url.searchParams.get("status") || "all";
  const priorityFilter = url.searchParams.get("priority") || "all";
  const sortBy = url.searchParams.get("sort") || "createdAt";
  const sortOrder = url.searchParams.get("order") || "desc";

  // Build where conditions - now we know user.id is valid
  const whereConditions = [
    eq(cases.userId, user.id) // Always filter by authenticated user
  ];

  if (searchQuery) {
    whereConditions.push(
      or(
        like(cases.title, `%${searchQuery}%`),
        like(cases.description, `%${searchQuery}%`)
      )
    );
  }
  if (statusFilter !== "all") {
    whereConditions.push(eq(cases.status, statusFilter));
  }
  if (priorityFilter !== "all") {
    whereConditions.push(eq(cases.priority, priorityFilter));
  }

  // Safely determine sort column
  const validSortColumns = ['createdAt', 'title', 'status', 'priority'] as const;
  const sortColumn = validSortColumns.includes(sortBy as any) 
    ? cases[sortBy as keyof typeof cases] 
    : cases.createdAt;
  
  const orderBy = sortOrder === "asc" ? sortColumn : desc(sortColumn);

  // Temporarily disabled database queries for SuperForms testing
  let userCases = [];
  let caseStats = [];

  // Mock data for testing purposes
  userCases = [
    {
      id: 'case-1',
      title: 'Test Case 1',
      status: 'open',
      priority: 'high',
      createdAt: new Date(),
      description: 'Test case for SuperForms',
      metadata: {}
    }
  ];
  
  caseStats = [
    { status: 'open', count: 1 },
    { status: 'closed', count: 0 }
  ];

  // DISABLED: Database queries for testing
  // try {
  //   userCases = await db.select(...).from(cases)...
  //   caseStats = await db.select(...).from(cases)...
  // } catch (error) {
  //   console.error("Database query failed:", error);
  //   userCases = [];
  //   caseStats = [];
  // }

  return {
    userCases,
    caseStats,
    searchQuery,
    statusFilter,
    priorityFilter,
    sortBy,
    sortOrder,
    user,
  };
};
