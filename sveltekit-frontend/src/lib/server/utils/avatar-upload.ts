import { createHash } from "crypto";

import { existsSync, mkdirSync, writeFileSync, unlinkSync } from "fs";
import { join } from "path";

export interface UploadConfig {
  uploadDir: string;
  maxFileSize: number;
  allowedTypes: string[];
  allowedExtensions: string[];
}

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export interface UploadResult {
  success: boolean;
  filePath?: string;
  fileName?: string;
  url?: string;
  error?: string;
}

// Default configuration for avatar uploads
export const AVATAR_UPLOAD_CONFIG: UploadConfig = {
  uploadDir: "static/uploads/avatars",
  maxFileSize: 5 * 1024 * 1024, // 5MB
  allowedTypes: [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/svg+xml",
    "image/webp",
  ],
  allowedExtensions: ["jpg", "jpeg", "png", "gif", "svg", "webp"],
};

/**
 * Comprehensive file validation for avatar uploads
 */
export function validateAvatarFile(
  file: File,
  config = AVATAR_UPLOAD_CONFIG,
): ValidationResult {
  // Check if file exists
  if (!file) {
    return { valid: false, error: "No file provided" };
  }

  // Check file size
  if (file.size === 0) {
    return { valid: false, error: "File is empty" };
  }

  if (file.size > config.maxFileSize) {
    const maxSizeMB = Math.round(config.maxFileSize / (1024 * 1024));
    return {
      valid: false,
      error: `File too large. Maximum size is ${maxSizeMB}MB`,
    };
  }

  // Check MIME type
  if (!config.allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: "Invalid file type. Allowed types: JPEG, PNG, GIF, SVG, WebP",
    };
  }

  // Check file extension
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!extension || !config.allowedExtensions.includes(extension)) {
    return {
      valid: false,
      error:
        "Invalid file extension. Allowed extensions: " +
        config.allowedExtensions.join(", "),
    };
  }

  // Check for potential security issues
  if (
    file.name.includes("..") ||
    file.name.includes("/") ||
    file.name.includes("\\")
  ) {
    return {
      valid: false,
      error: "Invalid file name",
    };
  }

  return { valid: true };
}

/**
 * Generate a secure, unique filename for avatar
 */
export function generateAvatarFileName(
  userId: string,
  originalFileName: string,
): string {
  const extension = originalFileName.split(".").pop()?.toLowerCase() || "jpg";
  const timestamp = Date.now();
  const randomHash = createHash("md5")
    .update(`${userId}-${timestamp}-${Math.random()}`)
    .digest("hex")
    .substring(0, 8);

  return `avatar_${userId}_${timestamp}_${randomHash}.${extension}`;
}

/**
 * Ensure upload directory exists with proper permissions
 */
export function ensureUploadDirectory(uploadDir: string): void {
  if (!existsSync(uploadDir)) {
    mkdirSync(uploadDir, { recursive: true, mode: 0o755 });
  }
}

/**
 * Handle avatar file upload with comprehensive error handling
 */
export async function handleAvatarUpload(
  file: File,
  userId: string,
  config = AVATAR_UPLOAD_CONFIG,
): Promise<UploadResult> {
  try {
    // Validate file
    const validation = validateAvatarFile(file, config);
    if (!validation.valid) {
      return { success: false, error: validation.error };
    }

    // Ensure upload directory exists
    ensureUploadDirectory(config.uploadDir);

    // Generate secure filename
    const fileName = generateAvatarFileName(userId, file.name);
    const filePath = join(config.uploadDir, fileName);

    // Convert file to buffer and save
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Additional security check - verify file is actually an image
    if (!isValidImageBuffer(buffer, file.type)) {
      return {
        success: false,
        error: "File content does not match declared type",
      };
    }

    // Write file to disk
    writeFileSync(filePath, buffer, { mode: 0o644 });

    // Generate public URL
    const url = `/uploads/avatars/${fileName}`;

    return {
      success: true,
      filePath,
      fileName,
      url,
    };
  } catch (error: any) {
    console.error("Avatar upload error:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Upload failed",
    };
  }
}

/**
 * Remove old avatar file from filesystem
 */
export function removeAvatarFile(avatarUrl: string | null): boolean {
  if (!avatarUrl || avatarUrl === "/images/default-avatar.svg") {
    return true; // Nothing to remove
  }

  try {
    // Extract filename from URL
    const fileName = avatarUrl.split("/").pop();
    if (!fileName) return false;

    const filePath = join(AVATAR_UPLOAD_CONFIG.uploadDir, fileName);

    if (existsSync(filePath)) {
      unlinkSync(filePath);
    }

    return true;
  } catch (error: any) {
    console.error("Error removing avatar file:", error);
    return false;
  }
}

/**
 * Basic image validation using file headers
 */
function isValidImageBuffer(buffer: Buffer, declaredType: string): boolean {
  const signatures: Record<string, number[]> = {
    "image/jpeg": [0xff, 0xd8, 0xff],
    "image/png": [0x89, 0x50, 0x4e, 0x47],
    "image/gif": [0x47, 0x49, 0x46],
    "image/webp": [0x52, 0x49, 0x46, 0x46],
  };

  // For SVG, check if it starts with valid XML/SVG tags
  if (declaredType === "image/svg+xml") {
    const content = buffer.toString("utf8", 0, 100).toLowerCase();
    return content.includes("<svg") || content.includes("<?xml");
  }

  const signature = signatures[declaredType];
  if (!signature) return true; // Allow unknown types to pass through

  // Check if buffer starts with expected signature
  for (let i = 0; i < signature.length; i++) {
    if (buffer[i] !== signature[i]) {
      return false;
    }
  }

  return true;
}

/**
 * Get file size in human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

/**
 * Generate initials from user name for avatar fallback
 */
export function generateInitials(user: {
  name?: string;
  firstName?: string;
  lastName?: string;
  email: string;
}): string {
  if (user.firstName && user.lastName) {
    return `${user.firstName.charAt(0)}${user.lastName.charAt(0)}`.toUpperCase();
  }

  if (user.name) {
    const nameParts = user.name.trim().split(" ");
    if (nameParts.length >= 2) {
      return `${nameParts[0].charAt(0)}${nameParts[nameParts.length - 1].charAt(0)}`.toUpperCase();
    }
    return nameParts[0].charAt(0).toUpperCase();
  }

  return user.email.charAt(0).toUpperCase();
}
