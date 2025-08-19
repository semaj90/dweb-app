// @ts-nocheck
// Registration API endpoint using Lucia v3 and bcrypt
import { users } from "$lib/server/db/schema-postgres";
type { RequestHandler }, {
json } from "@sveltejs/kit";
// Orphaned content: import { eq
import {
hashPassword } from "$lib/auth/password";
// Orphaned content: import { lucia
import {
db } from "$lib/server/db/index";

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const body = await request.json();
    const { name, email, password } = body;
    console.log("[Register API] Received data:", {
      name,
      email,
      password: "***",
    });

    // Validate input
    if (!email || !password || !name) {
      return json({ error: "Missing required fields" }, { status: 400 });
    }
    if (
      typeof email !== "string" ||
      typeof password !== "string" ||
      typeof name !== "string"
    ) {
      return json({ error: "Invalid input format" }, { status: 400 });
    }
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return json({ error: "Invalid email format" }, { status: 400 });
    }
    // Validate password strength
    if (password.length < 6) {
      return json(
        { error: "Password must be at least 6 characters long" },
        { status: 400 },
      );
    }
    // Check if user already exists
    const existingUsers = await db
      .select()
      .from(users)
      .where(eq(users.email, email.toLowerCase()))
      .limit(1);

    if (existingUsers.length > 0) {
      return json({ error: "User already exists" }, { status: 400 });
    }
    // Hash password
    const hashedPassword = await hashPassword(password);

    // Create user
    const [newUser] = await db
      .insert(users)
      .values({
        email: email.toLowerCase(),
        hashedPassword,
        name,
        firstName: name.split(" ")[0] || "",
        lastName: name.split(" ").slice(1).join(" ") || "",
        role: "prosecutor",
        isActive: true,
      })
      .returning();

    console.log("[Register API] User created successfully:", newUser.id);

    // Optionally create a session immediately after registration
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
      message: "Registration successful",
      user: {
        ...userInfo,
        avatarUrl: userInfo.avatarUrl || "/images/default-avatar.svg",
      },
    });
  } catch (error) {
    console.error("[Register API] Error:", error);
    return json(
      { error: "An error occurred during registration" },
      { status: 500 },
    );
  }
};
