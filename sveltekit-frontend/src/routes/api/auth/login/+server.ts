import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";
import { authService } from "$lib/server/auth";
import { isValidEmail } from "$lib/utils";

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const body = await request.json();
    const { email, password } = body;
    
    console.log("🔐 Login attempt for:", email);

    // Validate input
    if (!email || !password) {
      return json(
        { error: "Email and password are required" },
        { status: 400 }
      );
    }

    if (typeof email !== "string" || typeof password !== "string") {
      return json({ error: "Invalid input format" }, { status: 400 });
    }

    if (!isValidEmail(email)) {
      return json({ error: "Invalid email format" }, { status: 400 });
    }

    // Login user using our auth service
    const user = await authService.login(email.toLowerCase().trim(), password);

    // Create session
    const session = await authService.createSession(user.id);

    // Set session cookie
    cookies.set('session', session.id, {
      path: '/',
      httpOnly: true,
      sameSite: 'strict',
      secure: process.env.NODE_ENV === 'production',
      maxAge: 60 * 60 * 24 * 30 // 30 days
    });

    console.log("✅ Login successful for:", user.email);

    // Return user info (excluding sensitive data)
    return json({
      success: true,
      message: "Login successful",
      user: {
        id: user.id,
        email: user.email,
        displayName: user.displayName,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        legalSpecialties: user.legalSpecialties,
        preferences: user.preferences,
        emailVerified: user.emailVerified,
        createdAt: user.createdAt
      }
    });

  } catch (error) {
    console.error("❌ Login error:", error);

    // Handle specific errors
    if (error instanceof Error) {
      if (error.message.includes('Invalid email or password') || 
          error.message.includes('Account is deactivated') ||
          error.message.includes('Account is temporarily locked')) {
        return json(
          { error: error.message },
          { status: 401 }
        );
      }
    }

    return json(
      { error: "An error occurred during login" },
      { status: 500 }
    );
  }
};
