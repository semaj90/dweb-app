import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ locals }) => {
  try {
    const session = locals.session;
    const user = locals.user;

    if (!session) {
      return json({ user: null }, { status: 401 });
    }
    return json({
      user: {
        id: user.id,
        email: user.email,
        name: (user as any).name || (user as any).firstName || user.email.split('@')[0],
        role: user.role,
      },
    });
  } catch (error) {
    console.error("Auth me error:", error);
    return json({ user: null }, { status: 401 });
  }
};
