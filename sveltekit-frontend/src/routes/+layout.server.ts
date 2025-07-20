import { loginSchema, registerSchema } from "$lib/schemas";
import { superValidate } from "sveltekit-superforms";
import { zod } from "sveltekit-superforms/adapters";

// Expose session and user to all layouts/pages (SSR)
export const load = async ({ locals }) => {
  const loginForm = await superValidate(zod(loginSchema));
  const registerForm = await superValidate(zod(registerSchema));

  return {
    user: locals.user,
    loginForm,
    registerForm,
  };
};
