// Admin Layout Load Function
// Handles server-side authentication and authorization for admin routes

import type { LayoutLoad } from './$types';
import { redirect } from '@sveltejs/kit';
import { AccessControl } from '$lib/auth/roles';

export const load: LayoutLoad = async ({ fetch, url, depends }) => {
  depends('app:auth');
  
  try {
    // Check current session
    const sessionResponse = await fetch('/api/auth/session', {
      credentials: 'include'
    });
    
    if (!sessionResponse.ok) {
      throw redirect(302, `/login?redirect=${encodeURIComponent(url.pathname)}`);
    }
    
    const sessionData = await sessionResponse.json();
    
    if (!sessionData.user || !sessionData.session) {
      throw redirect(302, `/login?redirect=${encodeURIComponent(url.pathname)}`);
    }
    
    // Check if user has admin panel access
    const hasAdminAccess = AccessControl.hasPermission(
      sessionData.user.role, 
      'access_admin_panel'
    );
    
    if (!hasAdminAccess) {
      throw redirect(302, '/unauthorized');
    }
    
    return {
      user: sessionData.user,
      session: sessionData.session,
      permissions: AccessControl.getRolePermissions(sessionData.user.role)
    };
    
  } catch (error) {
    if (error instanceof Response) {
      throw error; // Re-throw redirects
    }
    
    console.error('Admin layout load error:', error);
    throw redirect(302, '/login');
  }
};