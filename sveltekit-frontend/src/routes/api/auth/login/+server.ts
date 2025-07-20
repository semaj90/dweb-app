// Login API endpoint using Lucia v3 and bcrypt
import { users } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from "@sveltejs/kit";
import { json } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { hashPassword, verifyPassword } from "$lib/auth/password";
import { lucia } from "$lib/auth/session";
import { db } from "$lib/server/db/index";

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const body = await request.json();
    const { email, password } = body;
    console.log("[Login API] Attempting login for:", email);

    // Validate input
    if (!email || !password) {
      return json(
        { error: "Email and password are required" },
        { status: 400 },
      );
    }
    if (typeof email !== "string" || typeof password !== "string") {
      return json({ error: "Invalid input format" }, { status: 400 });
    }
    // Demo user credentials for automatic creation
    const demoUsers = [
      {
        email: "admin@example.com",
        password: "admin123",
        name: "Demo Admin",
        role: "admin",
      },
      {
        email: "admin@prosecutor.com",
        password: "password",
        name: "Demo Admin",
        role: "admin",
      },
      {
        email: "user@example.com",
        password: "user123",
        name: "Demo User",
        role: "prosecutor",
      },
    ];

    // Find user by email
    const existingUsers = await db
      .select()
      .from(users)
      .where(eq(users.email, email.toLowerCase()))
      .limit(1);
    let user = existingUsers[0] || null;

    // If user doesn't exist and it's a demo user, create them
    if (!user) {
      const demoUser = demoUsers.find((du) => du.email === email);
      if (demoUser) {
        console.log("[Login API] Creating demo user:", email);
        const hashedPassword = await hashPassword(demoUser.password);

        const [newUser] = await db
          .insert(users)
          .values({
            email: demoUser.email,
            hashedPassword,
            name: demoUser.name,
            firstName: demoUser.name.split(" ")[0] || "",
            lastName: demoUser.name.split(" ").slice(1).join(" ") || "",
            role: demoUser.role,
            isActive: true,
          })
          .returning();

        user = newUser;
        console.log("[Login API] Demo user created:", user.id);
      }
    }
    if (!user || !user.hashedPassword) {
      return json({ error: "Invalid credentials" }, { status: 401 });
    }
    // Verify password
    const validPassword = await verifyPassword(password, user.hashedPassword);
    if (!validPassword) {
      return json({ error: "Invalid credentials" }, { status: 401 });
    }
    // Check if user is active
    if (!user.isActive) {
      return json({ error: "Account is deactivated" }, { status: 403 });
    }
    console.log("[Login API] Login successful for:", user.email);

    // Create Lucia session
    const session = await lucia.createSession(user.id, {});
    const sessionCookie = lucia.createSessionCookie(session.id);

    // Set the session cookie
    cookies.set(sessionCookie.name, sessionCookie.value, {
      path: ".",
      ...sessionCookie.attributes,
    });

    // Return user info (excluding password)
    const { hashedPassword, ...userInfo } = user;

    return json({
      success: true,
      message: "Login successful",
      user: {
        ...userInfo,
        avatarUrl: userInfo.avatarUrl || "/images/default-avatar.svg",
      },
    });
  } catch (err) {
    console.error("Login error:", err);
    return json({ error: "An error occurred during login" }, { status: 500 });
  }
};
