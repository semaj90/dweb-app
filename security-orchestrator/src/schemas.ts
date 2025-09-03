import { z } from 'zod';

export const deviceFingerprintSchema = z.object({
  userAgent: z.string(),
  // Basic IP (v4) pattern; adjust for IPv6 as needed
  ip: z.string().regex(/^(\d{1,3}\.){3}\d{1,3}$/,{ message: 'Invalid IPv4' }).optional(),
  timezone: z.string().optional(),
  language: z.string().optional(),
  screen: z.object({
    width: z.number().int().positive().optional(),
    height: z.number().int().positive().optional(),
    colorDepth: z.number().int().optional()
  }).optional(),
  platform: z.string().optional(),
  gpu: z.string().optional(),
  webglVendor: z.string().optional(),
  webglRenderer: z.string().optional(),
  canvasHash: z.string().optional(),
  fontsHash: z.string().optional(),
  pluginsHash: z.string().optional(),
  audioHash: z.string().optional(),
  mediaDevices: z.number().int().optional(),
  touchPoints: z.number().int().optional(),
  hardwareConcurrency: z.number().int().optional(),
  deviceMemory: z.number().int().optional(),
  localStorage: z.boolean().optional(),
  sessionStorage: z.boolean().optional(),
  timezoneOffset: z.number().optional(),
  secureContext: z.boolean().optional(),
});

export const userDataSchema = z.object({
  email: z.string().email(),
  username: z.string().min(3),
  requestedRole: z.string().optional(),
  referralCode: z.string().optional(),
  geo: z.object({
    country: z.string().optional(),
    region: z.string().optional(),
    city: z.string().optional(),
  }).optional(),
  behavioral: z.object({
    mouseMovementEntropy: z.number().min(0).max(1).optional(),
    keypressVariance: z.number().min(0).max(1).optional(),
    interactionLatencyMs: z.number().optional(),
  }).optional()
});

export const securityValidationRequestSchema = z.object({
  task: z.literal('security_validation'),
  fingerprint: deviceFingerprintSchema,
  user: userDataSchema,
  context: z.object({
    attempt: z.number().int().positive().default(1),
    ipReputation: z.number().min(0).max(1).optional(),
    velocity: z.number().min(0).max(1).optional(),
    knownDevice: z.boolean().optional(),
    previousFailures: z.number().int().optional(),
  }).optional()
});

export type SecurityValidationRequest = z.infer<typeof securityValidationRequestSchema>;

export const securityValidationResponseSchema = z.object({
  requestId: z.string().uuid(),
  riskScore: z.number().min(0).max(1),
  securityScore: z.number().min(0).max(1),
  verification: z.object({
    emailFormatValid: z.boolean(),
    usernameValid: z.boolean(),
    roleAllowed: z.boolean().optional(),
    referralValid: z.boolean().optional()
  }),
  signals: z.array(z.object({
    name: z.string(),
    weight: z.number(),
    value: z.any(),
    contribution: z.number()
  })),
  status: z.enum(['allow','review','deny']),
  modelVersion: z.string(),
  durationMs: z.number(),
  timestamp: z.string()
});

export type SecurityValidationResponse = z.infer<typeof securityValidationResponseSchema>;
