// Auth service stubs
export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}
export interface RegisterData extends LoginCredentials {
  name: string;
  role?: string;
}
export async function login(_c: LoginCredentials) {
  throw new Error("Not implemented");
}
export async function logout() {
  throw new Error("Not implemented");
}
export async function register(_d: RegisterData) {
  throw new Error("Not implemented");
}
