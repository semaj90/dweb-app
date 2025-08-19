import { registerSchema } from "$lib/schemas/auth";
// Orphaned content: import {

import { users } from "$lib/server/db/schema-postgres";
// Orphaned content: import {

import { fail, redirect } from "@sveltejs/kit";
// Orphaned content: import {

import type { JSONSchema7 } from "json-schema";
// Orphaned content: import {
message, superValidate
import { zod } from "sveltekit-superforms/adapters";
// Orphaned content: import type { Actions, PageServerLoad

export const load: PageServerLoad = async ({ locals }) => {
  if (locals.user) {
    throw redirect(303, "/dashboard");
  }

  const form = await superValidate(zod(registerSchema), {
    id: "register",
    jsonSchema: {
      type: "object",
      properties: {
        email: { type: "string" },
        password: { type: "string" },
        confirmPassword: { type: "string" },
        name: { type: "string" },
        role: { type: "string" },
        terms: { type: "boolean" },
      },
      required: ["email", "password", "confirmPassword", "name", "role"],
    } as JSONSchema7,
  });
  return { form };
};

export const actions: Actions = {
  default: async ({ request }) => {
    const form = await superValidate(request, zod(registerSchema), {
      id: "register",
      jsonSchema: {
        type: "object",
        properties: {
          email: { type: "string" },
          password: { type: "string" },
          confirmPassword: { type: "string" },
          name: { type: "string" },
          role: { type: "string" },
          terms: { type: "boolean" },
        },
        required: ["email", "password", "confirmPassword", "name", "role"],
      } as JSONSchema7,
    });

    if (!form.valid) {
      return fail(400, { form });
    }

    // Extra server-side constraints not encoded in schema
    if (form.data.password !== form.data.confirmPassword) {
      return message(form, "Passwords do not match", { status: 400 });
    }
    if (!form.data.terms) {
      return message(form, "You must accept the terms", { status: 400 });
    }
    try {
      const existingUser = await db
        .select({ id: users.id })
        .from(users)
        .where(eq(users.email, form.data.email as string))
        .limit(1);

      if (existingUser.length > 0) {
        return message(form, "An account with this email already exists.", {
          status: 400,
        });
      }

      const hashedPassword = await hash(form.data.password, {
        memoryCost: 19456,
        timeCost: 2,
        outputLen: 32,
        parallelism: 1,
      });

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
      throw redirect(302, "/login?registered=true");
    } catch (error) {
      console.error("[Register] Error:", error);
      if (error instanceof Response) throw error;
      return message(form, "Registration failed. Please try again.", {
        status: 500,
      });
    }
  },
};
