import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

const JWT_SECRET = process.env.JWT_SECRET || "fallback-secret-key";
const JWT_EXPIRATION = process.env.JWT_EXPIRATION || "86400"; // 24 hours

/**
 * Hashes a plain-text password using Bcrypt.
 * @param password The plain-text password.
 * @returns A promise that resolves to the hashed password.
 */
export async function hashPassword(password: string): Promise<string> {
  const rounds = parseInt(process.env.BCRYPT_ROUNDS || "12");
  return bcrypt.hash(password, rounds);
}
/**
 * Verifies a plain-text password against a stored hash.
 * @param password The plain-text password to verify.
 * @param hashedPassword The stored hash to compare against.
 * @returns A promise that resolves to true if the password is valid, otherwise false.
 */
export async function verifyPassword(
  password: string,
  hashed: string,
): Promise<boolean> {
  return bcrypt.compare(password, hashed);
}
/**
 * Signs a JWT token with the given payload.
 */
export function signJWT(payload: object): string {
  return jwt.sign(payload, JWT_SECRET, {
    expiresIn: "7d", // Use string format for expiration
  });
}
/**
 * Verifies a JWT token and returns the payload.
 */
export function verifyJWT(token: string): any {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (error) {
    return null;
  }
}
/**
 * Generates a secure random token.
 */
export function generateSecureToken(): string {
  return crypto.randomUUID();
}
