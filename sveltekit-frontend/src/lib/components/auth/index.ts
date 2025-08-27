/**
 * Authentication Components Export
 * User authentication and authorization components
 */

export { default as AuthForm } from './AuthForm.svelte';

export type AuthMode = 'login' | 'register' | 'forgot-password' | 'reset-password';
export type AuthProvider = 'email' | 'google' | 'github' | 'microsoft';

export interface AuthFormData {
  email: string;
  password?: string;
  confirmPassword?: string;
  firstName?: string;
  lastName?: string;
}