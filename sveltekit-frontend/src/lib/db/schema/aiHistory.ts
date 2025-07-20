import { pgTable, text, timestamp, uuid, vector } from "drizzle-orm/pg-core";
export const aiHistory = pgTable("ai_history", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: text("user_id"),
  prompt: text("prompt"),
  response: text("response"),
  embedding: vector("embedding", { dimensions: 768 }),
  createdAt: timestamp("created_at").defaultNow(),
});
