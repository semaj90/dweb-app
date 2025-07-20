import postgres from "postgres";
import dotenv from "dotenv";

dotenv.config();

const DATABASE_URL =
  process.env.DATABASE_URL ||
  "postgresql://postgres:postgres@localhost:5432/prosecutor_db";

async function verifyData() {
  const sql = postgres(DATABASE_URL);

  try {
    console.log("🔍 Verifying database data...\n");

    // Check users
    const users = await sql`SELECT id, email, role FROM users`;
    console.log(`👥 Users found: ${users.length}`);
    users.forEach((user) => {
      console.log(`  - ${user.email} (${user.role})`);
    });

    // Check cases
    const cases = await sql`SELECT id, title, status FROM cases`;
    console.log(`\n📋 Cases found: ${cases.length}`);
    cases.forEach((c) => {
      console.log(`  - ${c.title} (${c.status})`);
    });

    // Check evidence
    const evidence =
      await sql`SELECT id, evidence_type, description FROM evidence`;
    console.log(`\n🔍 Evidence found: ${evidence.length}`);
    evidence.forEach((e) => {
      console.log(
        `  - ${e.evidence_type}: ${e.description ? e.description.substring(0, 50) + "..." : "No description"}`,
      );
    });

    // Check reports
    const reports = await sql`SELECT id, title, report_type FROM reports`;
    console.log(`\n📄 Reports found: ${reports.length}`);
    reports.forEach((r) => {
      console.log(`  - ${r.title} (${r.report_type})`);
    });

    console.log("\n✅ Data verification complete!");
  } catch (error) {
    console.error("❌ Error verifying data:", error);
  } finally {
    await sql.end();
  }
}

verifyData();
