import type { RequestHandler } from '@sveltejs/kit';
import { json } from "@sveltejs/kit";
import { z } from "zod";
import { authService } from "$lib/server/auth";
import { embeddingService } from "$lib/server/embedding-service";
import { isValidEmail } from "$lib/utils";

const registerSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(8, "Password must be at least 8 characters"),
  firstName: z.string().optional(),
  lastName: z.string().optional(),
  displayName: z.string().optional(),
  legalSpecialties: z.array(z.string()).optional()
});

export const POST: RequestHandler = async ({ request, cookies }) => {
  try {
    const body = await request.json();
    console.log("📝 Registration attempt:", { email: body.email });

    // Validate input
    const validationResult = registerSchema.safeParse(body);
    if (!validationResult.success) {
      return json({
        success: false,
        error: "Invalid input data",
        details: validationResult.error.flatten()
      }, { status: 400 });
    }

    const { email, password, firstName, lastName, displayName, legalSpecialties } = validationResult.data;

    // Additional email validation
    if (!isValidEmail(email)) {
      return json({
        success: false,
        error: "Invalid email format"
      }, { status: 400 });
    }

    // Register user using our auth service
    const user = await authService.register({
      email: email.toLowerCase().trim(),
      password,
      firstName: firstName?.trim(),
      lastName: lastName?.trim(),
      displayName: displayName?.trim(),
      legalSpecialties: legalSpecialties || []
    });

    // Generate embeddings for RAG functionality
    try {
      await Promise.all([
        embeddingService.generateUserProfileEmbedding(user.id),
        embeddingService.generateUserPreferenceEmbedding(user.id)
      ]);
    } catch (embeddingError) {
      console.warn('Failed to generate user embeddings:', embeddingError);
      // Don't fail registration if embeddings fail
    }

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

    console.log("✅ User registered successfully:", {
      id: user.id,
      email: user.email
    });

    return json({
      success: true,
      message: "User registered successfully",
      user: {
        id: user.id,
        email: user.email,
        displayName: user.displayName,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        legalSpecialties: user.legalSpecialties,
        createdAt: user.createdAt
      }
    }, { status: 201 });

  } catch (error) {
    console.error("❌ Registration error:", error);

    // Handle specific errors
    if (error instanceof Error) {
      if (error.message === 'User already exists') {
        return json({
          success: false,
          error: "An account with this email already exists"
        }, { status: 409 });
      }
    }

    return json({
      success: false,
      error: "Registration failed. Please try again."
    }, { status: 500 });
  }
};