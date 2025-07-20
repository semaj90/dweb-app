// System Health Check for Enhanced Legal AI
// Monitors all services and provides real-time status

import fetch from "node-fetch";
import { Ollama } from "ollama";
import neo4j from "neo4j-driver";
import postgres from "postgres";

class HealthChecker {
  constructor() {
    this.services = {
      ollama: {
        url: "http://localhost:11434",
        name: "Ollama (Gemma 3 Legal AI)",
      },
      neo4j: { url: "bolt://localhost:7687", name: "Neo4j Knowledge Graph" },
      postgresql: {
        url: "postgresql://postgres:postgres@localhost:5432/prosecutor_db",
        name: "PostgreSQL Database",
      },
      qdrant: { url: "http://localhost:6333", name: "Qdrant Vector Search" },
      redis: { url: "redis://localhost:6379", name: "Redis Cache" },
      elasticsearch: { url: "http://localhost:9200", name: "Elasticsearch" },
      pgadmin: { url: "http://localhost:5050", name: "PgAdmin" },
    };

    this.healthStatus = {};
  }

  async checkOllama() {
    try {
      const ollama = new Ollama({ host: this.services.ollama.url });
      const models = await ollama.list();

      const hasGemma3 = models.models.some((m) =>
        m.name.includes("gemma3-legal-ai"),
      );
      const hasNomic = models.models.some((m) =>
        m.name.includes("nomic-embed"),
      );

      return {
        status: "healthy",
        details: {
          models_available: models.models.length,
          gemma3_legal_ai: hasGemma3,
          nomic_embeddings: hasNomic,
          model_list: models.models.map((m) => m.name).slice(0, 3),
        },
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        details: { service: "unreachable" },
      };
    }
  }

  async checkNeo4j() {
    try {
      const driver = neo4j.driver(
        this.services.neo4j.url,
        neo4j.auth.basic("neo4j", "prosecutorpassword"),
      );

      const session = driver.session();
      const result = await session.run(`
                MATCH (n) 
                RETURN labels(n) as labels, count(n) as count 
                ORDER BY count DESC 
                LIMIT 5
            `);

      const nodeStats = result.records.map((record) => ({
        label: record.get("labels")[0] || "Unknown",
        count: record.get("count").toNumber(),
      }));

      await session.close();
      await driver.close();

      return {
        status: "healthy",
        details: {
          total_nodes: nodeStats.reduce((sum, stat) => sum + stat.count, 0),
          node_types: nodeStats,
          knowledge_graph_initialized: nodeStats.length > 0,
        },
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        details: { connection: "failed" },
      };
    }
  }

  async checkPostgreSQL() {
    try {
      const sql = postgres(this.services.postgresql.url);

      // Check basic connectivity and database info
      const dbInfo = await sql`
                SELECT 
                    current_database() as database,
                    version() as version,
                    pg_database_size(current_database()) as size_bytes
            `;

      // Check enhanced tables
      const tables = await sql`
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN (
                    'users', 'cases', 'evidence', 'ai_recommendations', 
                    'document_embeddings', 'investigations', 'canvas_data',
                    'user_behavior', 'knowledge_entities'
                )
                ORDER BY table_name
            `;

      // Check pgvector extension
      const extensions = await sql`
                SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm')
            `;

      // Check record counts
      const userCount = await sql`SELECT count(*) as count FROM users`;
      const caseCount = await sql`SELECT count(*) as count FROM cases`;
      const evidenceCount = await sql`SELECT count(*) as count FROM evidence`;

      await sql.end();

      return {
        status: "healthy",
        details: {
          database: dbInfo[0].database,
          enhanced_tables: tables.map((t) => t.table_name),
          extensions: extensions.map((e) => e.extname),
          record_counts: {
            users: parseInt(userCount[0].count),
            cases: parseInt(caseCount[0].count),
            evidence: parseInt(evidenceCount[0].count),
          },
          size_mb: Math.round(parseInt(dbInfo[0].size_bytes) / 1024 / 1024),
        },
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        details: { connection: "failed" },
      };
    }
  }

  async checkQdrant() {
    try {
      const healthResponse = await fetch(`${this.services.qdrant.url}/health`);
      if (!healthResponse.ok) throw new Error(`HTTP ${healthResponse.status}`);

      const collectionsResponse = await fetch(
        `${this.services.qdrant.url}/collections`,
      );
      const collections = collectionsResponse.ok
        ? await collectionsResponse.json()
        : null;

      return {
        status: "healthy",
        details: {
          collections: collections?.result?.collections?.length || 0,
          collection_names:
            collections?.result?.collections?.map((c) => c.name) || [],
        },
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        details: { service: "unreachable" },
      };
    }
  }

  async checkElasticsearch() {
    try {
      const healthResponse = await fetch(
        `${this.services.elasticsearch.url}/_health`,
      );
      const clusterResponse = await fetch(
        `${this.services.elasticsearch.url}/`,
      );

      const health = healthResponse.ok ? await healthResponse.json() : null;
      const cluster = clusterResponse.ok ? await clusterResponse.json() : null;

      return {
        status: healthResponse.ok ? "healthy" : "degraded",
        details: {
          cluster_status: health?.status || "unknown",
          version: cluster?.version?.number || "unknown",
          cluster_name: cluster?.cluster_name || "unknown",
        },
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        details: { service: "unreachable" },
      };
    }
  }

  async checkHttpService(serviceName) {
    try {
      const service = this.services[serviceName];
      const response = await fetch(service.url, {
        method: "HEAD",
        timeout: 5000,
      });

      return {
        status: response.ok ? "healthy" : "degraded",
        details: {
          http_status: response.status,
          response_time: "sub_5s",
        },
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        details: { service: "unreachable" },
      };
    }
  }

  async runFullHealthCheck() {
    console.log("🏥 Enhanced Legal AI System Health Check");
    console.log("=" * 50);
    console.log(`🕐 ${new Date().toISOString()}\n`);

    const checks = [
      { name: "ollama", checker: () => this.checkOllama() },
      { name: "neo4j", checker: () => this.checkNeo4j() },
      { name: "postgresql", checker: () => this.checkPostgreSQL() },
      { name: "qdrant", checker: () => this.checkQdrant() },
      { name: "elasticsearch", checker: () => this.checkElasticsearch() },
      { name: "pgadmin", checker: () => this.checkHttpService("pgadmin") },
    ];

    const results = {};

    for (const check of checks) {
      try {
        console.log(`🔍 Checking ${this.services[check.name].name}...`);
        results[check.name] = await check.checker();

        const status = results[check.name].status;
        const emoji =
          status === "healthy" ? "✅" : status === "degraded" ? "⚠️" : "❌";
        console.log(
          `${emoji} ${this.services[check.name].name}: ${status.toUpperCase()}`,
        );

        if (results[check.name].details) {
          Object.entries(results[check.name].details).forEach(
            ([key, value]) => {
              console.log(`   📊 ${key}: ${JSON.stringify(value)}`);
            },
          );
        }

        if (results[check.name].error) {
          console.log(`   ❌ Error: ${results[check.name].error}`);
        }

        console.log();
      } catch (error) {
        results[check.name] = {
          status: "unhealthy",
          error: error.message,
        };
        console.log(`❌ ${this.services[check.name].name}: FAILED`);
        console.log(`   ❌ Error: ${error.message}\n`);
      }
    }

    // Summary
    const healthyCount = Object.values(results).filter(
      (r) => r.status === "healthy",
    ).length;
    const degradedCount = Object.values(results).filter(
      (r) => r.status === "degraded",
    ).length;
    const unhealthyCount = Object.values(results).filter(
      (r) => r.status === "unhealthy",
    ).length;
    const totalServices = Object.keys(results).length;

    console.log("📊 SYSTEM HEALTH SUMMARY");
    console.log("=" * 30);
    console.log(`✅ Healthy: ${healthyCount}/${totalServices}`);
    console.log(`⚠️  Degraded: ${degradedCount}/${totalServices}`);
    console.log(`❌ Unhealthy: ${unhealthyCount}/${totalServices}`);

    const overallHealth =
      healthyCount === totalServices
        ? "healthy"
        : healthyCount >= totalServices * 0.7
          ? "degraded"
          : "unhealthy";

    console.log(`\n🎯 Overall System Status: ${overallHealth.toUpperCase()}`);

    if (overallHealth === "healthy") {
      console.log(
        "🎉 All systems operational! Enhanced Legal AI ready for use.",
      );
    } else if (overallHealth === "degraded") {
      console.log(
        "⚠️  Some services need attention, but core functionality available.",
      );
    } else {
      console.log(
        "❌ Critical issues detected. System may not function properly.",
      );
    }

    // Recommendations
    console.log("\n🔧 RECOMMENDATIONS:");
    if (results.ollama?.status !== "healthy") {
      console.log("  🤖 Check Ollama service and model availability");
    }
    if (results.neo4j?.status !== "healthy") {
      console.log("  🕸️  Initialize Neo4j knowledge graph database");
    }
    if (results.postgresql?.status !== "healthy") {
      console.log("  📊 Verify PostgreSQL connection and schema setup");
    }
    if (healthyCount < totalServices) {
      console.log(
        "  🐳 Consider restarting Docker services: docker compose -f ../docker-compose.enhanced.yml restart",
      );
    }

    return {
      overall: overallHealth,
      services: results,
      summary: {
        healthy: healthyCount,
        degraded: degradedCount,
        unhealthy: unhealthyCount,
        total: totalServices,
      },
    };
  }
}

// Export for use in other modules
export default HealthChecker;

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const checker = new HealthChecker();
  checker
    .runFullHealthCheck()
    .then((results) => {
      const exitCode = results.overall === "unhealthy" ? 1 : 0;
      process.exit(exitCode);
    })
    .catch((error) => {
      console.error("💥 Health check failed:", error);
      process.exit(1);
    });
}
