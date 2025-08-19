import { URL } from "url";
// @ts-nocheck
/**
 * Advanced Security Middleware
 * Rate Limiting, JWT Refresh, and Security Headers
 */

import { dev } from "$app/environment";
// Orphaned content: import {

import type { RequestEvent } from "@sveltejs/kit";
import { logWarn, logError, , interface RateLimitEntry {,   count: number;,   resetTime: number;,   blocked: boolean; } from
interface SecurityConfig {
  rateLimits: {
    general: { requests: number; windowMs: number };
    auth: { requests: number; windowMs: number };
    api: { requests: number; windowMs: number };
    upload: { requests: number; windowMs: number };
  };
  jwt: {
    accessTokenExpiry: string;
    refreshTokenExpiry: string;
  };
}
const config: SecurityConfig = {
  rateLimits: {
    general: { requests: 1000, windowMs: 15 * 60 * 1000 }, // 1000 requests per 15 minutes
    auth: { requests: 5, windowMs: 15 * 60 * 1000 }, // 5 auth attempts per 15 minutes
    api: { requests: 300, windowMs: 15 * 60 * 1000 }, // 300 API calls per 15 minutes
    upload: { requests: 10, windowMs: 60 * 1000 }, // 10 uploads per minute
  },
  jwt: {
    accessTokenExpiry: "15m",
    refreshTokenExpiry: "7d",
  },
};

class SecurityManager {
  private rateLimitStore = new Map<string, RateLimitEntry>();
  private blockedIPs = new Set<string>();

  /**
   * Rate limiting middleware
   */
  rateLimit(type: keyof SecurityConfig["rateLimits"] = "general") {
    return (event: RequestEvent) => {
      const ip = this.getClientIP(event);
      const key = `${ip}:${type}`;
      const limit = config.rateLimits[type];
      const now = Date.now();

      // Clean expired entries
      this.cleanExpiredEntries();

      // Check if IP is blocked
      if (this.blockedIPs.has(ip)) {
        logWarn("Blocked IP attempted request", {
          ip,
          endpoint: event.url.pathname,
        });
        throw error(429, "IP temporarily blocked due to abuse");
      }
      // Get or create rate limit entry
      let entry = this.rateLimitStore.get(key);
      if (!entry || now > entry.resetTime) {
        entry = {
          count: 0,
          resetTime: now + limit.windowMs,
          blocked: false,
        };
      }
      entry.count++;
      this.rateLimitStore.set(key, entry);

      // Check if limit exceeded
      if (entry.count > limit.requests) {
        entry.blocked = true;

        // Block IP temporarily for repeated abuse
        if (entry.count > limit.requests * 2) {
          this.blockedIPs.add(ip);
          setTimeout(() => this.blockedIPs.delete(ip), 60 * 60 * 1000); // 1 hour block
        }
        logWarn("Rate limit exceeded", {
          ip,
          type,
          count: entry.count,
          limit: limit.requests,
          endpoint: event.url.pathname,
        });

        throw error(
          429,
          `Rate limit exceeded. Retry after ${Math.ceil((entry.resetTime - now) / 1000)} seconds`,
        );
      }
      // Add rate limit headers
      event.setHeaders({
        "X-RateLimit-Limit": limit.requests.toString(),
        "X-RateLimit-Remaining": (limit.requests - entry.count).toString(),
        "X-RateLimit-Reset": entry.resetTime.toString(),
      });
    };
  }
  /**
   * Security headers middleware
   */
  securityHeaders() {
    return (event: RequestEvent) => {
      event.setHeaders({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": this.getCSP(),
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
      });
    };
  }
  /**
   * JWT token refresh logic
   */
  async refreshToken(refreshToken: string) {
    try {
      // Verify refresh token
      const payload = await this.verifyJWT(refreshToken);

      if (!payload || payload.type !== "refresh") {
        throw new Error("Invalid refresh token");
      }
      // Generate new access token
      const newAccessToken = await this.generateJWT(
        {
          userId: payload.userId,
          email: payload.email,
          role: payload.role,
          type: "access",
        },
        config.jwt.accessTokenExpiry,
      );

      return {
        accessToken: newAccessToken,
        expiresIn: 15 * 60, // 15 minutes
      };
    } catch (error) {
      logError("Token refresh failed", { error });
      throw error;
    }
  }
  /**
   * Audit logging for sensitive actions
   */
  auditLog(event: RequestEvent, action: string, metadata?: any) {
    const ip = this.getClientIP(event);
    const userAgent = event.request.headers.get("user-agent");

    logWarn("Security audit log", {
      action,
      ip,
      userAgent,
      endpoint: event.url.pathname,
      timestamp: new Date().toISOString(),
      metadata,
    });
  }
  /**
   * Content Security Policy
   */
  private getCSP(): string {
    const nonce = this.generateNonce();

    if (dev) {
      return `default-src 'self'; script-src 'self' 'unsafe-eval' 'nonce-${nonce}'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' ws: wss:`;
    }
    return `default-src 'self'; script-src 'self' 'nonce-${nonce}'; style-src 'self' 'nonce-${nonce}'; img-src 'self' data:; connect-src 'self'; base-uri 'self'; form-action 'self'`;
  }
  private generateNonce(): string {
    return Math.random().toString(36).substring(2, 15);
  }
  private getClientIP(event: RequestEvent): string {
    const forwarded = event.request.headers.get("x-forwarded-for");
    const realIP = event.request.headers.get("x-real-ip");
    const remoteAddr = event.getClientAddress();

    return (
      forwarded?.split(",")[0]?.trim() || realIP || remoteAddr || "unknown"
    );
  }
  private cleanExpiredEntries() {
    const now = Date.now();
    for (const [key, entry] of this.rateLimitStore.entries()) {
      if (now > entry.resetTime) {
        this.rateLimitStore.delete(key);
      }
    }
  }
  private async verifyJWT(token: string): Promise<any> {
    // Implement JWT verification logic
    // This is a placeholder - use your JWT library
    try {
      // const jwt = require('jsonwebtoken');
      // return jwt.verify(token, import.meta.env.JWT_SECRET);
      return null; // Placeholder
    } catch (error) {
      return null;
    }
  }
  private async generateJWT(payload: any, expiresIn: string): Promise<string> {
    // Implement JWT generation logic
    // This is a placeholder - use your JWT library
    try {
      // const jwt = require('jsonwebtoken');
      // return jwt.sign(payload, import.meta.env.JWT_SECRET, { expiresIn });
      return "placeholder-token"; // Placeholder
    } catch (error) {
      throw error;
    }
  }
}
// Export singleton instance
export const security = new SecurityManager();

// Middleware helpers
export const rateLimitGeneral = () => security.rateLimit("general");
export const rateLimitAuth = () => security.rateLimit("auth");
export const rateLimitAPI = () => security.rateLimit("api");
export const rateLimitUpload = () => security.rateLimit("upload");
export const addSecurityHeaders = () => security.securityHeaders();
