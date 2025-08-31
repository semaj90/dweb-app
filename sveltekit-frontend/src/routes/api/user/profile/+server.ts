
import { users } from "$lib/server/db/index";
import { eq } from 'drizzle-orm';
import { apiResponse, transformUserForFrontend } from '$lib/utils/case-transform';

import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ locals }) => {
  const authUser = locals.user;
  if (!authUser?.id) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    // Query using actual database column names (snake_case)
    const user = await db.query.users.findFirst({
      where: eq(users.id, locals.user.id),
      columns: {
        id: true,
        email: true,
        username: true,          // database field
        first_name: true,        // database field
        last_name: true,         // database field  
        role: true,
        avatar_url: true,        // database field
        created_at: true,        // database field
        updated_at: true,        // database field
        is_active: true,         // database field
        email_verified: true,    // database field
      },
    });

    if (!user) {
      return json({ error: "User not found" }, { status: 404 });
    }
    // Transform snake_case database fields to camelCase for frontend
    const frontendUser = transformUserForFrontend(user);
    
    return json({
      success: true,
      user: {
        ...frontendUser,
        avatarUrl: frontendUser.avatarUrl || "/images/default-avatar.png",
      },
    });
  } catch (error: any) {
    console.error("Profile fetch error:", error);
    return json({ error: "Failed to fetch profile" }, { status: 500 });
  }
};

export const PUT: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    // Frontend sends camelCase data
    const frontendData = await request.json();
    const { firstName, lastName, email } = frontendData;

    if (!email) {
      return json({ error: "Email is required" }, { status: 400 });
    }

    // Update user profile in database using snake_case fields
    const [updatedUser] = await db
      .update(users)
      .set({
        email,
        first_name: firstName || "",      // snake_case for database
        last_name: lastName || "",        // snake_case for database  
        updated_at: new Date(),           // snake_case for database
      })
      .where(eq(users.id, locals.user.id))
      .returning({
        id: users.id,
        email: users.email,
        username: users.username,
        first_name: users.first_name,     // snake_case database field
        last_name: users.last_name,       // snake_case database field
        role: users.role,
        avatar_url: users.avatar_url,     // snake_case database field
        created_at: users.created_at,     // snake_case database field
        updated_at: users.updated_at,     // snake_case database field
      });

    if (!updatedUser) {
      return json({ error: "Failed to update profile" }, { status: 500 });
    }

    // Transform snake_case database result to camelCase for frontend
    const frontendUser = transformUserForFrontend(updatedUser);
    
    return json({
      success: true,
      user: {
        ...frontendUser,
        avatarUrl: frontendUser.avatarUrl || "/images/default-avatar.svg",
      },
      message: "Profile updated successfully",
    });
  } catch (error: any) {
    console.error("Profile update error:", error);
    return json({ error: "Failed to update profile" }, { status: 500 });
  }
};
