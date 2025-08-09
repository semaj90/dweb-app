// @ts-nocheck
import { users } from "$lib/server/db/schema-postgres";
import { fail, redirect } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";
import { hash } from "@node-rs/argon2";
import { registerSchema } from "$lib/schemas";
import { superValidate, message } from "sveltekit-superforms";
import { zod } from "sveltekit-superforms/adapters";
import type { Actions, PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ locals }) => {
  // If user is already logged in, redirect to dashboard
  if (locals.user) {
    throw redirect(303, "/dashboard");
  }
  
  const form = await superValidate(zod(registerSchema));
  return { form };
};

export const actions: Actions = {
  default: async ({ request }) => {
    const form = await superValidate(request, zod(registerSchema));
    
    if (!form.valid) {
      return fail(400, { form });
    }
    try {
      // Check if user already exists
      const existingUser = await db
        .select({ id: users.id })
        .from(users)
        .where(eq(users.email, form.data.email))
        .limit(1);

      if (existingUser.length > 0) {
        return message(form, "An account with this email already exists.", { status: 400 });
      }

      // Hash password
      const hashedPassword = await hash(form.data.password, {
        memoryCost: 19456,
        timeCost: 2,
        outputLen: 32,
        parallelism: 1,
      });

      // Create user
      const [newUser] = await db
        .insert(users)
        .values({
          email: form.data.email,
          hashedPassword,
          name: form.data.name,
          firstName: form.data.name.split(" ")[0] || "",
          lastName: form.data.name.split(" ").slice(1).join(" ") || "",
          role: form.data.role,
          isActive: true,
        })
        .returning();

      console.log("[Register] User created successfully:", newUser.id);

      // Redirect to login page with success message
      throw redirect(302, "/login?registered=true");
    } catch (error) {
      console.error("[Register] Error:", error);

      // If it's a redirect, re-throw it
      if (error instanceof Response) {
        throw error;
      }
      
      return message(form, "Registration failed. Please try again.", { status: 500 });
    }
  },
};
