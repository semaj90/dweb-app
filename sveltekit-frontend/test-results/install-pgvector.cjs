const { Client } = require("pg");

async function installPgvector() {
  console.log("🔧 Installing pgvector extension...");

  const client = new Client({
    user: "postgres",
    host: "localhost",
    database: "prosecutor_db",
    password: "postgres",
    port: 5433,
  });

  try {
    await client.connect();
    console.log("✅ Connected to PostgreSQL");

    // Install pgvector extension
    await client.query("CREATE EXTENSION IF NOT EXISTS vector");
    console.log("✅ pgvector extension installed successfully!");

    // Verify installation
    const result = await client.query(
      "SELECT extname FROM pg_extension WHERE extname = 'vector'",
    );
    if (result.rows.length > 0) {
      console.log("✅ pgvector extension verified");
    } else {
      console.log("❌ pgvector extension installation failed");
    }
  } catch (error) {
    console.error("❌ Failed to install pgvector:", error.message);
  } finally {
    await client.end();
  }
}

// Run the installation
installPgvector().catch(console.error);
