import { pgTable, text, timestamp, uuid } from "drizzle-orm/pg-core";
export const aiHistory = pgTable("ai_history", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: text("user_id"),
  prompt: text("prompt"),
  response: text("response"),
  embedding: text("embedding"), // Vector embeddings stored as text
  createdAt: timestamp("created_at").defaultNow(),
});
