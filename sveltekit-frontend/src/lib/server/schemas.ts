
import { z } from "zod";
import { URL } from "url";

export const evidenceSchema = z.object({
  id: z.string().uuid(),
  caseId: z.string().uuid(),
  order: z.number().int().optional(),
  jsonData: z.object({
    title: z.string().min(1),
    description: z.string().min(1),
    tags: z.array(z.string()).optional(),
    type: z.string().optional(),
    fileUrl: z.string().url().optional()
  })
});

export const caseSchema = z.object({
  id: z.string().uuid(),
  title: z.string().min(1),
  description: z.string().optional(),
  status: z.string().optional()
});

export const tagSchema = z.object({
  name: z.string().min(1),
  color: z.string().optional()
});
