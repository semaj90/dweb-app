import { lucia } from "$lib/auth/session";
import { loginSchema } from "$lib/schemas/auth";
import { db } from "$lib/server/db/index";
import { users } from "$lib/server/db/schema-postgres";
import { verify } from "@node-rs/argon2";
import { fail, redirect } from "@sveltejs/kit";
import bcrypt from "bcrypt";
import { eq } from "drizzle-orm";
import { message, superValidate } from "sveltekit-superforms";
import type { Actions, PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ locals, url }) => {
  // If user is already logged in, redirect to dashboard
  if (locals.user) {
    throw redirect(303, "/dashboard");
  }

  // Provide a SuperValidated form for SSR compatibility
  const form = await superValidate(loginSchema, {
    id: "login",
    jsonSchema: false,
  });

  // Registration success banner
  const registered = url.searchParams.get("registered");
  const registrationSuccess =
    registered === "true"
      ? "Account created successfully! You can now sign in."
      : undefined;

  return { form, registrationSuccess };
};

export const actions: Actions = {
  default: async ({ request, cookies }) => {
    // Validate incoming request using superforms
    const form = await superValidate(request, loginSchema, {
      id: "login",
      jsonSchema: false,
    });

    if (!form.valid) {
      return fail(400, { form });
    }

    const { email, password } = form.data;

    try {
      // Find user by email
      const existingUser = await db
        .select()
        .from(users)
        .where(eq(users.email, email))
        .limit(1);

      if (!existingUser.length || !existingUser[0].hashedPassword) {
        return message(form, "Incorrect email or password", { status: 400 });
      }

      const user = existingUser[0];

      // Check if user is active
      if (!user.isActive) {
        return message(form, "Account is deactivated", { status: 403 });
      }

      // Verify password - try both bcrypt and argon2 for compatibility
      let validPassword = false;
      try {
        validPassword = await bcrypt.compare(password, user.hashedPassword);
      } catch {
        try {
          validPassword = await verify(user.hashedPassword, password);
        } catch (error) {
          console.error("Password verification failed:", error);
        }
      }

      if (!validPassword) {
        return message(form, "Incorrect email or password", { status: 400 });
      }

      // Create Lucia session
      const session = await lucia.createSession(user.id, {});
      const sessionCookie = lucia.createSessionCookie(session.id);

      cookies.set(sessionCookie.name, sessionCookie.value, {
        path: ".",
        ...sessionCookie.attributes,
      });

      console.log(`[Login] User ${user.email} logged in successfully`);
      throw redirect(303, "/dashboard");
    } catch (error) {
      console.error("[Login] Error:", error);
      if (error instanceof Response) throw error;
      return message(form, "Login failed. Please try again.", { status: 500 });
    }
  },
};
