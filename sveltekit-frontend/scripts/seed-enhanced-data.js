// Enhanced Database Seeding Script
// Seeds the database with comprehensive legal AI sample data
// Optimized for low memory development environment

import postgres from "postgres";
import { Ollama } from "ollama";
import { createHash } from "crypto";

const sql = postgres(
  "postgresql://postgres:postgres@localhost:5432/prosecutor_db",
);
const ollama = new Ollama({ host: "http://localhost:11434" });

console.log("🌱 Enhanced Legal AI Database Seeding");
console.log("=" * 50);

async function generateEmbedding(text, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await ollama.embeddings({
        model: "nomic-embed-text",
        prompt: text,
      });
      return response.embedding;
    } catch (error) {
      console.log(`  ⚠️  Embedding attempt ${i + 1} failed: ${error.message}`);
      if (i === retries - 1) {
        // Return null embedding as fallback
        console.log("  📝 Using null embedding as fallback");
        return null;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
}

async function seedEnhancedData() {
  try {
    console.log("\n📊 Checking current database state...");

    // Check if data already exists
    const userCount = await sql`SELECT count(*) as count FROM users`;
    const caseCount = await sql`SELECT count(*) as count FROM cases`;
    const evidenceCount = await sql`SELECT count(*) as count FROM evidence`;

    console.log(
      `  📋 Current state: ${userCount[0].count} users, ${caseCount[0].count} cases, ${evidenceCount[0].count} evidence`,
    );

    if (
      parseInt(userCount[0].count) >= 5 &&
      parseInt(caseCount[0].count) >= 5
    ) {
      console.log("  ✅ Sample data already exists, skipping basic seed");
    } else {
      console.log("  🌱 Basic sample data will be created by SQL seed script");
    }

    // Generate embeddings for existing cases and evidence
    console.log("\n🔗 Generating vector embeddings...");

    // Get cases without embeddings
    const casesNeedingEmbeddings = await sql`
            SELECT id, title, description, content 
            FROM cases 
            WHERE title_embedding IS NULL 
            LIMIT 10
        `;

    console.log(
      `  📝 Processing embeddings for ${casesNeedingEmbeddings.length} cases...`,
    );

    for (const case_row of casesNeedingEmbeddings) {
      try {
        console.log(
          `    🔍 Processing case: ${case_row.title.substring(0, 50)}...`,
        );

        // Generate title embedding
        const titleEmbedding = await generateEmbedding(case_row.title);

        // Generate content embedding (use description if content is too long)
        const contentText =
          case_row.content || case_row.description || case_row.title;
        const truncatedContent = contentText.substring(0, 500); // Limit for memory
        const contentEmbedding = await generateEmbedding(truncatedContent);

        // Update case with embeddings
        if (titleEmbedding && contentEmbedding) {
          await sql`
                        UPDATE cases 
                        SET title_embedding = ${titleEmbedding}::vector,
                            content_embedding = ${contentEmbedding}::vector
                        WHERE id = ${case_row.id}
                    `;
          console.log(
            `    ✅ Embeddings added for case: ${case_row.title.substring(0, 30)}...`,
          );
        } else {
          console.log(
            `    ⚠️  Skipped embeddings for case: ${case_row.title.substring(0, 30)}...`,
          );
        }

        // Small delay to prevent overwhelming the embedding service
        await new Promise((resolve) => setTimeout(resolve, 500));
      } catch (error) {
        console.log(
          `    ❌ Error processing case ${case_row.id}: ${error.message}`,
        );
        continue;
      }
    }

    // Get evidence without embeddings
    const evidenceNeedingEmbeddings = await sql`
            SELECT id, filename, extracted_text 
            FROM evidence 
            WHERE content_embedding IS NULL 
            AND extracted_text IS NOT NULL
            LIMIT 10
        `;

    console.log(
      `  📁 Processing embeddings for ${evidenceNeedingEmbeddings.length} evidence items...`,
    );

    for (const evidence of evidenceNeedingEmbeddings) {
      try {
        console.log(
          `    📄 Processing evidence: ${evidence.filename.substring(0, 40)}...`,
        );

        // Use first 500 characters of extracted text
        const textContent = evidence.extracted_text.substring(0, 500);
        const embedding = await generateEmbedding(textContent);

        if (embedding) {
          await sql`
                        UPDATE evidence 
                        SET content_embedding = ${embedding}::vector
                        WHERE id = ${evidence.id}
                    `;
          console.log(
            `    ✅ Embedding added for evidence: ${evidence.filename.substring(0, 30)}...`,
          );
        } else {
          console.log(
            `    ⚠️  Skipped embedding for evidence: ${evidence.filename.substring(0, 30)}...`,
          );
        }

        await new Promise((resolve) => setTimeout(resolve, 300));
      } catch (error) {
        console.log(
          `    ❌ Error processing evidence ${evidence.id}: ${error.message}`,
        );
        continue;
      }
    }

    // Create document embeddings table entries
    console.log("\n📚 Creating document embedding entries...");

    const casesWithEmbeddings = await sql`
            SELECT id, title, content, title_embedding, content_embedding
            FROM cases 
            WHERE title_embedding IS NOT NULL 
            LIMIT 5
        `;

    for (const case_row of casesWithEmbeddings) {
      try {
        // Insert title embedding
        await sql`
                    INSERT INTO document_embeddings (entity_type, entity_id, embedding_type, embedding, model_name, chunk_text, metadata)
                    VALUES (
                        'case',
                        ${case_row.id},
                        'title',
                        ${case_row.title_embedding},
                        'nomic-embed-text',
                        ${case_row.title},
                        ${'{"source": "case_title", "processing_method": "automated"}'}
                    )
                    ON CONFLICT DO NOTHING
                `;

        // Insert content embedding
        if (case_row.content_embedding) {
          const contentText = (case_row.content || case_row.title).substring(
            0,
            500,
          );
          await sql`
                        INSERT INTO document_embeddings (entity_type, entity_id, embedding_type, embedding, model_name, chunk_text, metadata)
                        VALUES (
                            'case',
                            ${case_row.id},
                            'content',
                            ${case_row.content_embedding},
                            'nomic-embed-text',
                            ${contentText},
                            ${'{"source": "case_content", "processing_method": "automated"}'}
                        )
                        ON CONFLICT DO NOTHING
                    `;
        }

        console.log(
          `    ✅ Document embeddings created for case: ${case_row.title.substring(0, 30)}...`,
        );
      } catch (error) {
        console.log(
          `    ❌ Error creating document embeddings: ${error.message}`,
        );
      }
    }

    // Add sample AI recommendations based on existing data
    console.log("\n🎯 Creating AI recommendations...");

    const users = await sql`SELECT id, role FROM users LIMIT 5`;
    const cases = await sql`SELECT id, case_type, user_id FROM cases LIMIT 5`;

    const recommendations = [
      {
        user_id: users[0]?.id,
        type: "similar_case",
        entity_type: "case",
        entity_id: cases[0]?.id,
        score: 0.85,
        reasoning: "Similar case type and evidence patterns detected",
      },
      {
        user_id: users[1]?.id,
        type: "expert_witness",
        entity_type: "case",
        entity_id: cases[1]?.id,
        score: 0.92,
        reasoning: "Forensic expert recommended based on evidence type",
      },
      {
        user_id: users[2]?.id,
        type: "research_topic",
        entity_type: "case",
        entity_id: cases[2]?.id,
        score: 0.78,
        reasoning: "Recent case law updates available for this practice area",
      },
    ];

    for (const rec of recommendations) {
      if (rec.user_id && rec.entity_id) {
        try {
          await sql`
                        INSERT INTO ai_recommendations (
                            user_id, recommendation_type, entity_type, entity_id, 
                            score, reasoning, context, expires_at, is_active
                        ) VALUES (
                            ${rec.user_id},
                            ${rec.type},
                            ${rec.entity_type},
                            ${rec.entity_id},
                            ${rec.score},
                            ${rec.reasoning},
                            ${"{}"},
                            ${new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)}, -- 30 days
                            true
                        )
                        ON CONFLICT DO NOTHING
                    `;
          console.log(`    ✅ Recommendation created: ${rec.type} for user`);
        } catch (error) {
          console.log(
            `    ⚠️  Recommendation creation skipped: ${error.message}`,
          );
        }
      }
    }

    // Create sample user behavior entries
    console.log("\n👤 Creating user behavior tracking data...");

    const behaviorActions = [
      { action: "view", entity_type: "case", duration: 15000 },
      { action: "search", entity_type: "evidence", duration: 3000 },
      { action: "edit", entity_type: "case", duration: 8000 },
      { action: "upload", entity_type: "evidence", duration: 12000 },
      { action: "ai_query", entity_type: "assistant", duration: 25000 },
    ];

    for (const user of users.slice(0, 3)) {
      for (const action of behaviorActions.slice(0, 3)) {
        try {
          await sql`
                        INSERT INTO user_behavior (
                            user_id, session_id, action_type, entity_type, 
                            context, timestamp, duration_ms, success
                        ) VALUES (
                            ${user.id},
                            ${"demo_session_" + Math.random().toString(36).substr(2, 9)},
                            ${action.action},
                            ${action.entity_type},
                            ${'{"demo": true, "source": "seed_script"}'},
                            ${new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000)},
                            ${action.duration},
                            true
                        )
                    `;
        } catch (error) {
          // Skip conflicts silently
        }
      }
    }

    console.log("    ✅ User behavior data created");

    // Final statistics
    console.log("\n📊 Final database statistics:");
    const finalStats = await sql`
            SELECT 
                (SELECT count(*) FROM users) as users,
                (SELECT count(*) FROM cases) as cases,
                (SELECT count(*) FROM evidence) as evidence,
                (SELECT count(*) FROM investigations) as investigations,
                (SELECT count(*) FROM ai_recommendations) as recommendations,
                (SELECT count(*) FROM user_behavior) as behavior_entries,
                (SELECT count(*) FROM document_embeddings) as embeddings,
                (SELECT count(*) FROM canvas_data) as canvas_items
        `;

    const stats = finalStats[0];
    console.log(`  👥 Users: ${stats.users}`);
    console.log(`  📁 Cases: ${stats.cases}`);
    console.log(`  📄 Evidence: ${stats.evidence}`);
    console.log(`  🔍 Investigations: ${stats.investigations}`);
    console.log(`  🎯 AI Recommendations: ${stats.recommendations}`);
    console.log(`  👤 Behavior Entries: ${stats.behavior_entries}`);
    console.log(`  🔗 Document Embeddings: ${stats.embeddings}`);
    console.log(`  🎨 Canvas Items: ${stats.canvas_items}`);

    console.log("\n🎉 Enhanced database seeding completed successfully!");
    console.log("\n💡 You can now:");
    console.log("  🔐 Login with: prosecutor@legalai.demo / demo password");
    console.log("  📁 View sample cases with evidence");
    console.log("  🔍 Try detective mode investigations");
    console.log("  🎨 Use interactive canvas features");
    console.log("  🤖 Ask the Legal AI assistant questions");
    console.log("  📊 Explore vector search and recommendations");
  } catch (error) {
    console.error("❌ Seeding error:", error);
    throw error;
  } finally {
    await sql.end();
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  seedEnhancedData()
    .then(() => {
      console.log("\n✅ Seeding script completed successfully");
      process.exit(0);
    })
    .catch((error) => {
      console.error("\n💥 Seeding script failed:", error);
      process.exit(1);
    });
}

export default seedEnhancedData;
