import { loginSchema } from "$lib/schemas";
import { lucia } from "$lib/server/auth";
import { db, users } from "$lib/server/db/index";
import { verify } from "@node-rs/argon2";
import { fail, redirect } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { message, superValidate } from "sveltekit-superforms";
import { zod } from "sveltekit-superforms/adapters";
import type { Actions, PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ locals }) => {
  // If user is already logged in, redirect to dashboard
  if (locals.user) {
    throw redirect(303, "/dashboard");
  }
  const form = await superValidate(zod(loginSchema));
  return { form };
};

export const actions: Actions = {
  default: async ({ request, cookies }) => {
    const form = await superValidate(request, zod(loginSchema));
    if (!form.valid) return fail(400, { form });

    const existingUser = await db.query.users.findFirst({
      where: eq(users.email, form.data.email),
    });
    if (!existingUser || !existingUser.hashedPassword) {
      return message(form, "Incorrect email or password.", { status: 400 });
    }
    const validPassword = await verify(
      existingUser.hashedPassword,
      form.data.password,
    );
    if (!validPassword) {
      return message(form, "Incorrect email or password.", { status: 400 });
    }
    const session = await lucia.createSession(existingUser.id, {});
    const sessionCookie = lucia.createSessionCookie(session.id);
    cookies.set(sessionCookie.name, sessionCookie.value, {
      path: ".",
      ...sessionCookie.attributes,
    });

    throw redirect(303, "/dashboard");
  },
};
