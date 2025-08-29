import type { Case } from "$lib/types";


/**
 * Comprehensive validation utilities for the Detective Mode app
 * Provides type-safe validation, sanitization, and error handling
 */

// Basic validation types
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  value?: unknown;
}
export interface ValidationRule<T> {
  name: string;
  validate: (value: T) => ValidationResult | boolean;
  message?: string;
  severity?: "error" | "warning";
}
export interface FormFieldConfig {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  type?: "email" | "url" | "phone" | "text" | "number" | "date";
  custom?: ValidationRule<any>[];
}
// Core validation functions
export function createValidationResult(
  isValid: boolean,
  errors: string[] = [],
  warnings: string[] = [],
  value?: unknown,
): ValidationResult {
  return { isValid, errors, warnings, value };
}
export function validateField(
  value: any,
  config: FormFieldConfig,
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Required validation
  if (
    config.required &&
    (value === null || value === undefined || value === "")
  ) {
    errors.push("This field is required");
    return createValidationResult(false, errors, warnings);
  }
  // Skip other validations if value is empty and not required
  if (
    !config.required &&
    (value === null || value === undefined || value === "")
  ) {
    return createValidationResult(true, errors, warnings, value);
  }
  const stringValue = String(value);

  // Length validation
  if (config.minLength && stringValue.length < config.minLength) {
    errors.push(`Must be at least ${config.minLength} characters long`);
  }
  if (config.maxLength && stringValue.length > config.maxLength) {
    errors.push(`Must be no more than ${config.maxLength} characters long`);
  }
  // Pattern validation
  if (config.pattern && !config.pattern.test(stringValue)) {
    errors.push("Invalid format");
  }
  // Type validation
  switch (config.type) {
    case "email":
      if (!isValidEmail(stringValue)) {
        errors.push("Invalid email address");
      }
      break;
    case "url":
      if (!isValidURL(stringValue)) {
        errors.push("Invalid URL");
      }
      break;
    case "phone":
      if (!isValidPhone(stringValue)) {
        errors.push("Invalid phone number");
      }
      break;
    case "number":
      if (isNaN(Number(value))) {
        errors.push("Must be a valid number");
      }
      break;
    case "date":
      if (!isValidDate(stringValue)) {
        errors.push("Invalid date");
      }
      break;
  }
  // Custom validation rules
  if (config.custom) {
    for (const rule of config.custom) {
      const result = rule.validate(value);
      const validationResult =
        typeof result === "boolean"
          ? createValidationResult(
              result,
              result ? [] : [rule.message || `${rule.name} validation failed`],
            )
          : result;

      if (rule.severity === "warning") {
        warnings.push(...validationResult.errors);
      } else {
        errors.push(...validationResult.errors);
      }
    }
  }
  return createValidationResult(errors.length === 0, errors, warnings, value);
}
// Specific validation functions
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email.toLowerCase());
}
export function isValidURL(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}
export function isValidPhone(phone: string): boolean {
  const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
  const cleanPhone = phone.replace(/[\s\-\(\)\.]/g, "");
  return phoneRegex.test(cleanPhone);
}
export function isValidDate(date: string): boolean {
  const parsedDate = new Date(date);
  return !isNaN(parsedDate.getTime());
}
// File validation
export interface FileValidationConfig {
  maxSize?: number; // in bytes
  allowedTypes?: string[];
  allowedExtensions?: string[];
  requireHash?: boolean;
}
export function validateFile(
  file: File,
  config: FileValidationConfig,
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Size validation
  if (config.maxSize && file.size > config.maxSize) {
    errors.push(
      `File size must be less than ${formatFileSize(config.maxSize)}`,
    );
  }
  // Type validation
  if (config.allowedTypes && !config.allowedTypes.includes(file.type)) {
    errors.push(`File type ${file.type} is not allowed`);
  }
  // Extension validation
  if (config.allowedExtensions) {
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !config.allowedExtensions.includes(extension)) {
      errors.push(`File extension .${extension} is not allowed`);
    }
  }
  // Security checks
  if (isDangerousFile(file.name)) {
    errors.push("File type is potentially dangerous");
  }
  return createValidationResult(errors.length === 0, errors, warnings);
}
// Case data validation
export interface CaseValidationConfig {
  title: FormFieldConfig;
  description: FormFieldConfig;
  status: FormFieldConfig;
  priority: FormFieldConfig;
  assignedTo?: FormFieldConfig;
}
export function validateCaseData(data: any): ValidationResult {
  const config: CaseValidationConfig = {
    title: {
      required: true,
      minLength: 3,
      maxLength: 200,
      custom: [
        {
          name: "title-format",
          validate: (value: string) => !value.includes("<script>"),
          message: "Title contains invalid characters",
        },
      ],
    },
    description: {
      required: true,
      minLength: 10,
      maxLength: 5000,
    },
    status: {
      required: true,
      custom: [
        {
          name: "valid-status",
          validate: (value: string) =>
            ["Open", "In-Progress", "Closed", "Archived"].includes(value),
          message: "Invalid status value",
        },
      ],
    },
    priority: {
      required: true,
      custom: [
        {
          name: "valid-priority",
          validate: (value: string) =>
            ["High", "Medium", "Low"].includes(value),
          message: "Invalid priority value",
        },
      ],
    },
  };

  const errors: string[] = [];
  const warnings: string[] = [];

  for (const [field, fieldConfig] of Object.entries(config)) {
    const result = validateField(data[field], fieldConfig);
    errors.push(...result.errors.map((e: any) => `${field}: ${e}`));
    warnings.push(...result.warnings.map((w) => `${field}: ${w}`));
  }
  return createValidationResult(errors.length === 0, errors, warnings, data);
}
// Evidence validation
export function validateEvidenceData(data: any): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Basic required fields
  if (!data.title || data.title.trim().length < 3) {
    errors.push("Evidence title must be at least 3 characters");
  }
  if (!data.type) {
    errors.push("Evidence type is required");
  }
  if (!data.caseId) {
    errors.push("Case ID is required");
  }
  // File-specific validation
  if (data.type === "file" && !data.filePath && !data.fileUrl) {
    errors.push("File path or URL is required for file evidence");
  }
  // Hash validation
  if (data.hash && !isValidHash(data.hash)) {
    warnings.push("Hash format appears to be invalid");
  }
  return createValidationResult(errors.length === 0, errors, warnings, data);
}
// Utility functions
export function formatFileSize(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${Math.round(size * 100) / 100} ${units[unitIndex]}`;
}
export function isDangerousFile(filename: string): boolean {
  const dangerousExtensions = [
    "exe",
    "bat",
    "cmd",
    "com",
    "pif",
    "scr",
    "vbs",
    "js",
    "jar",
    "ps1",
    "sh",
    "py",
    "pl",
    "rb",
    "php",
    "asp",
    "aspx",
    "jsp",
  ];

  const extension = filename.split(".").pop()?.toLowerCase();
  return extension ? dangerousExtensions.includes(extension) : false;
}
export function isValidHash(hash: string): boolean {
  // Support common hash formats (MD5, SHA1, SHA256, SHA512)
  const md5Regex = /^[a-f0-9]{32}$/i;
  const sha1Regex = /^[a-f0-9]{40}$/i;
  const sha256Regex = /^[a-f0-9]{64}$/i;
  const sha512Regex = /^[a-f0-9]{128}$/i;

  return (
    md5Regex.test(hash) ||
    sha1Regex.test(hash) ||
    sha256Regex.test(hash) ||
    sha512Regex.test(hash)
  );
}
// Form validation utilities
export class FormValidator {
  private fields: Map<string, FormFieldConfig> = new Map();
  private values: Map<string, any> = new Map();
  private errors: Map<string, string[]> = new Map();
  private warnings: Map<string, string[]> = new Map();

  addField(name: string, config: FormFieldConfig): void {
    this.fields.set(name, config);
  }
  setValue(name: string, value: any): ValidationResult {
    this.values.set(name, value);

    const config = this.fields.get(name);
    if (!config) {
      return createValidationResult(true);
    }
    const result = validateField(value, config);

    if (result.errors.length > 0) {
      this.errors.set(name, result.errors);
    } else {
      this.errors.delete(name);
    }
    if (result.warnings.length > 0) {
      this.warnings.set(name, result.warnings);
    } else {
      this.warnings.delete(name);
    }
    return result;
  }
  validateAll(): ValidationResult {
    const allErrors: string[] = [];
    const allWarnings: string[] = [];

    for (const [name, config] of this.fields) {
      const value = this.values.get(name);
      const result = validateField(value, config);

      if (result.errors.length > 0) {
        allErrors.push(...result.errors.map((e: any) => `${name}: ${e}`));
        this.errors.set(name, result.errors);
      }
      if (result.warnings.length > 0) {
        allWarnings.push(...result.warnings.map((w) => `${name}: ${w}`));
        this.warnings.set(name, result.warnings);
      }
    }
    return createValidationResult(
      allErrors.length === 0,
      allErrors,
      allWarnings,
    );
  }
  getFieldErrors(name: string): string[] {
    return this.errors.get(name) || [];
  }
  getFieldWarnings(name: string): string[] {
    return this.warnings.get(name) || [];
  }
  hasErrors(): boolean {
    return this.errors.size > 0;
  }
  hasFieldError(name: string): boolean {
    return this.errors.has(name);
  }
  clear(): void {
    this.values.clear();
    this.errors.clear();
    this.warnings.clear();
  }
  getValues(): Record<string, any> {
    const result: Record<string, any> = {};
    for (const [key, value] of this.values) {
      result[key] = value;
    }
    return result;
  }
}
// Input sanitization
export function sanitizeInput(input: string): string {
  return input
    .replace(/[<>]/g, "") // Remove potential HTML tags
    .replace(/javascript:/gi, "") // Remove javascript: protocols
    .replace(/on\w+=/gi, "") // Remove event handlers
    .trim();
}
export function sanitizeFilename(filename: string): string {
  return filename
    .replace(/[^\w\s.-]/g, "") // Keep only alphanumeric, spaces, dots, and hyphens
    .replace(/\s+/g, "_") // Replace spaces with underscores
    .toLowerCase();
}
// Data structure validation
export function validateObject(
  obj: any,
  schema: Record<string, FormFieldConfig>,
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  for (const [key, config] of Object.entries(schema)) {
    const result = validateField(obj[key], config);
    errors.push(...result.errors.map((e: any) => `${key}: ${e}`));
    warnings.push(...result.warnings.map((w) => `${key}: ${w}`));
  }
  return createValidationResult(errors.length === 0, errors, warnings, obj);
}
