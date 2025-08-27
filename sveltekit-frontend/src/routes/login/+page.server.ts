import {
  verifyPassword,
  createUserSession,
  setSessionCookie
} from "$lib/server/lucia";
import { loginSchema } from "$lib/schemas/auth";
import { db } from "$lib/server/db/index";
import { users } from "$lib/server/db/schema-postgres";
import { fail, redirect } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import type { JSONSchema7 } from "json-schema";
// Temporarily disabled SuperForms due to Svelte 5 compatibility issue
// import { message, superValidate } from "sveltekit-superforms";
// import { zod } from "sveltekit-superforms/adapters";
import type { Actions, PageServerLoad } from "./$types";

// removed: import { URL } from "url";

export const load: PageServerLoad = async ({ locals, url }) => {
  const localsTyped = locals as unknown as App.Locals;

  // If user is already logged in, redirect to dashboard
  if (localsTyped.user) {
    throw redirect(303, "/dashboard");
  }

  // Registration success banner
  const registered = url.searchParams.get("registered");
  const registrationSuccess =
    registered === "true"
      ? "Account created successfully! You can now sign in."
      : undefined;

  return { registrationSuccess };
};

export const actions: Actions = {
  default: async ({ request, cookies }) => {
    const formData = await request.formData();
    const email = formData.get('email') as string;
    const password = formData.get('password') as string;

    // Basic validation
    if (!email || !password) {
      return fail(400, { error: 'Email and password are required' });
    }

    if (!email.includes('@')) {
      return fail(400, { error: 'Please enter a valid email address' });
    }
    const localsTyped = (globalThis as any).__CURRENT_LOCALS__ || ({} as App.Locals);

    try {
      // Find user by email
      const existingUser = await db
        .select()
        .from(users)
        .where(eq(users.email, email as string))
        .limit(1);

      if (!existingUser.length || !existingUser[0].hashedPassword) {
        return fail(400, { error: "Incorrect email or password" });
      }

      const user = existingUser[0];

      // Check if user is active
      if (!user.isActive) {
        return fail(403, { error: "Account is deactivated" });
      }

      // Verify password using custom lucia
      const validPassword = await verifyPassword(user.hashedPassword!, password as string);

      if (!validPassword) {
        console.log(`[Login] Password verification failed for ${user.email}`);
        return fail(400, { error: "Incorrect email or password" });
      }

      // Create session using custom lucia
      const { sessionId, expiresAt } = await createUserSession(user.id);
      setSessionCookie(cookies, sessionId, expiresAt);

      // Dev debug: print short session id to server logs for quick verification
      if (process.env.NODE_ENV === 'development') {
        console.log(`[Login] session set: ${sessionId.substring(0, 12)}... for ${user.email}`);
      }

      console.log(`[Login] User ${user.email} logged in successfully`);
      throw redirect(303, "/dashboard");
    } catch (error) {
      console.error("[Login] Error:", error);
      if (error instanceof Response) throw error;
      return fail(500, { error: "Login failed. Please try again." });
    }
  },
};
