import { drizzle } from "drizzle-orm/postgres-js";
import { migrate } from "drizzle-orm/postgres-js/migrator";
import postgres from "postgres";

const DATABASE_URL =
  process.env.DATABASE_URL ||
  "postgresql://legal_user:legal_pass_2024@localhost:5432/deeds_legal";

async function runMigrations() {
  console.log("🔄 Starting database migrations...");

  const migrationClient = postgres(DATABASE_URL, { max: 1 });
  const db = drizzle(migrationClient);

  try {
    // Enable pgvector extension
    console.log("📦 Enabling pgvector extension...");
    await migrationClient`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log("✅ pgvector extension enabled");

    // Run Drizzle migrations
    console.log("🗄️ Running table migrations...");
    await migrate(db, { migrationsFolder: "./drizzle" });
    console.log("✅ Database migrations completed");

    // Create indexes for better performance
    console.log("📊 Creating performance indexes...");

    // Vector similarity indexes
    await migrationClient`
      CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
      ON document_chunks USING ivfflat (embedding vector_cosine_ops)
      WITH (lists = 100)
    `;

    await migrationClient`
      CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx
      ON knowledge_base USING ivfflat (embedding vector_cosine_ops)
      WITH (lists = 100)
    `;

    // Regular indexes for common queries
    await migrationClient`CREATE INDEX IF NOT EXISTS messages_session_id_idx ON messages (session_id)`;
    await migrationClient`CREATE INDEX IF NOT EXISTS messages_timestamp_idx ON messages (timestamp)`;
    await migrationClient`CREATE INDEX IF NOT EXISTS chat_sessions_user_id_idx ON chat_sessions (user_id)`;
    await migrationClient`CREATE INDEX IF NOT EXISTS chat_sessions_case_id_idx ON chat_sessions (case_id)`;
    await migrationClient`CREATE INDEX IF NOT EXISTS documents_case_id_idx ON documents (case_id)`;
    await migrationClient`CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx ON document_chunks (document_id)`;
    await migrationClient`CREATE INDEX IF NOT EXISTS evidence_case_id_idx ON evidence (case_id)`;
    await migrationClient`CREATE INDEX IF NOT EXISTS search_queries_user_id_idx ON search_queries (user_id)`;

    console.log("✅ Performance indexes created");

    console.log("🎉 All migrations completed successfully!");
  } catch (error) {
    console.error("❌ Migration failed:", error);
    process.exit(1);
  } finally {
    await migrationClient.end();
  }
}

runMigrations();
