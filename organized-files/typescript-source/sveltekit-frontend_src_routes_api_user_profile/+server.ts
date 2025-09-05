// @ts-nocheck
import { users } from "$lib/server/db/index";
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";

export const GET: RequestHandler = async ({ locals }) => {
  if (!locals.user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    const user = await db.query.users.findFirst({
      where: eq(users.id, locals.user.id),
      columns: {
        id: true,
        email: true,
        name: true,
        firstName: true,
        lastName: true,
        role: true,
        avatarUrl: true,
        createdAt: true,
      },
    });

    if (!user) {
      return json({ error: "User not found" }, { status: 404 });
    }
    return json({
      success: true,
      user: {
        ...user,
        avatarUrl: user.avatarUrl || "/images/default-avatar.png",
      },
    });
  } catch (error) {
    console.error("Profile fetch error:", error);
    return json({ error: "Failed to fetch profile" }, { status: 500 });
  }
};

export const PUT: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    const { name, email, firstName, lastName } = await request.json();

    if (!email || !name) {
      return json({ error: "Name and email are required" }, { status: 400 });
    }
    // Update user profile in database
    const [updatedUser] = await db
      .update(users)
      .set({
        name,
        email,
        firstName: firstName || "",
        lastName: lastName || "",
      })
      .where(eq(users.id, locals.user.id))
      .returning({
        id: users.id,
        email: users.email,
        name: users.name,
        firstName: users.firstName,
        lastName: users.lastName,
        role: users.role,
        avatarUrl: users.avatarUrl,
        createdAt: users.createdAt,
      });

    if (!updatedUser) {
      return json({ error: "Failed to update profile" }, { status: 500 });
    }
    return json({
      success: true,
      user: {
        ...updatedUser,
        avatarUrl: updatedUser.avatarUrl || "/images/default-avatar.svg",
      },
      message: "Profile updated successfully",
    });
  } catch (error) {
    console.error("Profile update error:", error);
    return json({ error: "Failed to update profile" }, { status: 500 });
  }
};
