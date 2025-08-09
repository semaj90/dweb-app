// @ts-nocheck
import { loginSchema } from "$lib/schemas";
import { lucia } from "$lib/server/auth";
import { db } from "$lib/server/db/index";
import { users } from "$lib/server/db/schema-postgres";
import { verify } from "@node-rs/argon2";
import { fail, redirect } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { message, superValidate } from "sveltekit-superforms";
import { zod } from "sveltekit-superforms/adapters";
import type { Actions, PageServerLoad } from "./$types";
import bcrypt from "bcrypt";

export const load: PageServerLoad = async ({ locals, url }) => {
  // If user is already logged in, redirect to dashboard
  if (locals.user) {
    throw redirect(303, "/dashboard");
  }
  
  // Simple approach - create form directly without Superforms for now
  const form = {
    valid: true,
    data: { email: '', password: '' },
    errors: {}
  };
  
  // Check for registration success message
  const registered = url.searchParams.get('registered');
  if (registered === 'true') {
    return { 
      form, 
      registrationSuccess: 'Account created successfully! You can now sign in.' 
    };
  }
  
  return { form };
};

export const actions: Actions = {
  default: async ({ request, cookies }) => {
    const formData = await request.formData();
    const email = formData.get('email')?.toString();
    const password = formData.get('password')?.toString();
    
    if (!email || !password) {
      return fail(400, { 
        form: { 
          data: { email: email || '', password: '' }, 
          errors: { email: !email ? 'Email is required' : '', password: !password ? 'Password is required' : '' },
          message: 'Please fill in all fields'
        } 
      });
    }

    try {
      // Find user by email
      const existingUser = await db
        .select()
        .from(users)
        .where(eq(users.email, email))
        .limit(1);

      if (!existingUser.length || !existingUser[0].hashedPassword) {
        return fail(400, { 
          form: { 
            data: { email, password: '' }, 
            errors: {},
            message: 'Incorrect email or password'
          } 
        });
      }

      const user = existingUser[0];

      // Check if user is active
      if (!user.isActive) {
        return fail(403, { 
          form: { 
            data: { email, password: '' }, 
            errors: {},
            message: 'Account is deactivated'
          } 
        });
      }

      // Verify password - try both bcrypt and argon2 for compatibility
      let validPassword = false;
      
      try {
        // Try bcrypt first (for demo users)
        validPassword = await bcrypt.compare(password, user.hashedPassword);
      } catch {
        try {
          // Fallback to argon2 (for registered users)
          validPassword = await verify(user.hashedPassword, password);
        } catch (error) {
          console.error('Password verification failed:', error);
        }
      }

      if (!validPassword) {
        return fail(400, { 
          form: { 
            data: { email, password: '' }, 
            errors: {},
            message: 'Incorrect email or password'
          } 
        });
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
      console.error('[Login] Error:', error);
      
      // If it's a redirect, re-throw it
      if (error instanceof Response) {
        throw error;
      }
      
      return fail(500, { 
        form: { 
          data: { email, password: '' }, 
          errors: {},
          message: 'Login failed. Please try again.'
        } 
      });
    }
  },
};
