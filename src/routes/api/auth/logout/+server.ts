// POST /api/auth/logout
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { AuthService } from '$lib/yorha/services/auth.service';

export const POST: RequestHandler = async ({ cookies, locals }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    
    if (!sessionToken) {
      return json({
        success: false,
        error: 'No active session'
      }, { status: 401 });
    }
    
    const authService = new AuthService();
    
    // Validate session first
    const sessionData = await authService.validateSession(sessionToken);
    
    if (sessionData) {
      // Log out the user
      await authService.logout(sessionData.session.id);
    }
    
    // Clear session cookie
    cookies.delete('yorha_session', { path: '/' });
    
    return json({
      success: true,
      message: 'Logged out successfully'
    });
  } catch (error: unknown) {
    console.error('Logout error:', error);
    
    // Clear cookie anyway
    cookies.delete('yorha_session', { path: '/' });
    
    return json({
      success: false,
      error: 'Logout failed'
    }, { status: 500 });
  }
};