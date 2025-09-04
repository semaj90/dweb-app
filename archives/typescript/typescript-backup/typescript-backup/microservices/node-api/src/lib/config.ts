import { z } from 'zod';
const EnvSchema = z.object({
  NATS_URL: z.string().default('nats://127.0.0.1:4222'),
  DATABASE_URL: z.string().default('postgresql://postgres:postgres@localhost:5432/legal_ai_db'),
  REDIS_URL: z.string().optional(),
  NODE_API_SERVICE_NAME: z.string().default('node-api'),
  NODE_API_MAX_PAYLOAD_KB: z.coerce.number().default(256),
  NODE_API_SUBJECT_WHITELIST: z.string().optional(),
});
const parsed = EnvSchema.safeParse(process.env);
if(!parsed.success){
  console.error('Config validation failed', parsed.error.flatten().fieldErrors);
  process.exit(1);
}
export const config = parsed.data;
export function subjectAllowed(subject: string, whitelist: string[]){
  return whitelist.length === 0 || whitelist.includes(subject);
}
