import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";
import type { RequestHandler } from "@sveltejs/kit";
import { z } from "zod";
import { users } from "$lib/server/db/schema-postgres";
import { lucia } from "$lib/server/auth";
import { db } from "$lib/server/db";
import { eq } from "drizzle-orm";
import { Argon2id } from "oslo/password";

const registerSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(8, "Password must be at least 8 characters"),
  firstName: z.string().min(1, "First name is required"),
  lastName: z.string().min(1, "Last name is required"),
  role: z.enum(["prosecutor", "investigator", "admin", "analyst"]).default("prosecutor"),
  department: z.string().optional(),
  badgeNumber: z.string().optional(),
  jurisdiction: z.string().optional()
});

export const POST: RequestHandler = async ({ request, getClientAddress, cookies }) => {
  const startTime = Date.now();
  const ipAddress = getClientAddress();
  const userAgent = request.headers.get("user-agent") || "";

  try {
    const body = await request.json();
    console.log("📝 Registration attempt:", { email: body.email, role: body.role });

    // Validate input
    const validationResult = registerSchema.safeParse(body);
    if (!validationResult.success) {
      return json({
        success: false,
        error: "Invalid input data",
        details: validationResult.error.flatten()
      }, { status: 400 });
    }

    const { email, password, firstName, lastName, role, department, badgeNumber, jurisdiction } = validationResult.data;

    // Check if user already exists
    const existingUsers = await db
      .select()
      .from(users)
      .where(eq(users.email, email.toLowerCase()))
      .limit(1);

    if (existingUsers.length > 0) {
      return json({
        success: false,
        error: "User with this email already exists"
      }, { status: 409 });
    }

    // Hash password
    const hashedPassword = await new Argon2id().hash(password);

    // Create user in database
    const [newUser] = await db
      .insert(users)
      .values({
        email: email.toLowerCase(),
        hashedPassword,
        firstName,
        lastName,
        name: `${firstName} ${lastName}`,
        role,
        isActive: true,
      })
      .returning();

    console.log("✅ User registered successfully:", {
      id: newUser.id,
      email: newUser.email,
      role: newUser.role,
      processingTime: Date.now() - startTime
    });

    // Create session for new user
    const session = await lucia.createSession(newUser.id, {});
    const sessionCookie = lucia.createSessionCookie(session.id);

    // Set the session cookie
    cookies.set(sessionCookie.name, sessionCookie.value, {
      path: ".",
      ...sessionCookie.attributes,
    });

    // Return user info (excluding password)
    const { hashedPassword: _, ...userInfo } = newUser;

    return json({
      success: true,
      message: "User registered successfully",
      user: {
        ...userInfo,
        avatarUrl: userInfo.avatarUrl || "/images/default-avatar.svg",
      }
    }, { status: 201 });

  } catch (error) {
    console.error("❌ Registration error:", error);

    return json({
      success: false,
      error: "Internal server error during registration"
    }, { status: 500 });
  }
};

// Helper function for default permissions
function getDefaultPermissions(role: string): string[] {
  const permissions = {
    prosecutor: [
      "cases:read",
      "cases:create", 
      "cases:update",
      "evidence:read",
      "evidence:create",
      "criminals:read",
      "ai:analyze"
    ],
    investigator: [
      "cases:read",
      "evidence:read",
      "evidence:create",
      "evidence:update",
      "criminals:read",
      "criminals:create",
      "criminals:update"
    ],
    analyst: [
      "cases:read",
      "evidence:read",
      "criminals:read",
      "ai:analyze",
      "reports:create"
    ],
    admin: [
      "cases:*",
      "evidence:*", 
      "criminals:*",
      "users:*",
      "system:*",
      "ai:*"
    ]
  };

  return permissions[role as keyof typeof permissions] || permissions.prosecutor;
}