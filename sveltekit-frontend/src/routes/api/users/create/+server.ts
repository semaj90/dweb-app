import { db } from '$lib/server/db';
import { users } from '$lib/server/db/schema-postgres';
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Check if user is authenticated and has admin role
    if (!locals.user || locals.user.role !== "admin") {
      return json(
        { error: "Unauthorized - Admin access required" },
        { status: 403 },
      );
    }
    const userData = await request.json();

    // Validate required fields
    if (!userData.email || !userData.name) {
      return json(
        {
          error: "Email and name are required fields",
        },
        { status: 400 },
      );
    }
    // Create the user
    const newUser = {
      email: userData.email,
      name: userData.name,
      firstName: userData.firstName,
      lastName: userData.lastName,
      role: userData.role || "prosecutor",
      password: userData.password || "password123",
      avatarUrl: userData.avatarUrl,
    } as const;

    const result = await db.insert(users).values(newUser as any).returning();

    if (result.length > 0) {
      return json({
        success: true,
        message: "User created successfully",
        user: result[0],
      });
    } else {
      return json(
        {
          success: false,
          message: "Failed to create user",
          error: "Database insertion failed",
        },
        { status: 500 },
      );
    }
  } catch (error: any) {
    console.error("Error creating user:", error);
    return json(
      {
        success: false,
        message: "Failed to create user",
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

export const GET: RequestHandler = async ({ locals }) => {
  // Check if user is authenticated
  if (!locals.user) {
    return json({ error: "Unauthorized" }, { status: 401 });
  }
  return json({
    message: "User creation endpoint",
    usage: {
      method: "POST",
      endpoint: "/api/users/create",
      body: {
        email: "user@example.com",
        name: "Full Name",
        firstName: "First (optional)",
        lastName: "Last (optional)",
        role: "prosecutor|detective|analyst|admin (optional, defaults to prosecutor)",
        password: "password (optional, defaults to password123)",
        avatarUrl: "URL to avatar image (optional)",
      },
    },
    examples: [
      {
        description: "Create a prosecutor",
        body: {
          email: "prosecutor@example.com",
          name: "John Prosecutor",
          role: "prosecutor",
        },
      },
      {
        description: "Create a detective",
        body: {
          email: "detective@example.com",
          name: "Sarah Detective",
          role: "detective",
          password: "customPassword123",
        },
      },
      {
        description: "Create an analyst",
        body: {
          email: "analyst@example.com",
          name: "Mike Analyst",
          firstName: "Mike",
          lastName: "Analyst",
          role: "analyst",
        },
      },
    ],
  });
};
