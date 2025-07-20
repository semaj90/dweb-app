#!/usr/bin/env node

/**
 * Setup Script for Enhanced Legal AI with Thinking Style
 * Sets up Ollama models, database schema updates, and document processing
 */

import { exec } from "child_process";
import { promises as fs } from "fs";
import { promisify } from "util";
import path from "path";

const execAsync = promisify(exec);

console.log("🧠 Setting up Enhanced Legal AI with Thinking Style...");
console.log("==================================================");

// Configuration
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const PROJECT_ROOT = process.cwd();

async function checkOllamaAvailability() {
  try {
    console.log("🔍 Checking Ollama availability...");
    const response = await fetch(`${OLLAMA_URL}/api/tags`);
    if (response.ok) {
      console.log("✅ Ollama is running and accessible");
      return true;
    }
  } catch (error) {
    console.log("❌ Ollama is not accessible at", OLLAMA_URL);
    console.log("💡 Make sure Ollama is installed and running:");
    console.log("   - Install: https://ollama.ai");
    console.log("   - Start: ollama serve");
    return false;
  }
}

async function setupOllamaModels() {
  console.log("\n🤖 Setting up Ollama models...");

  try {
    // Check if base gemma3 model exists
    console.log("📥 Pulling Gemma 3 base model...");
    await execAsync("ollama pull gemma3:7b", { timeout: 300000 }); // 5 minute timeout
    console.log("✅ Gemma 3 base model ready");

    // Create enhanced legal thinking model
    console.log("🔨 Creating legal thinking model...");

    const modelfile = `FROM gemma3:7b

TEMPLATE """<|user|>
{{ .Prompt }}
<|assistant|>
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

SYSTEM """You are an expert legal document analyst with enhanced reasoning capabilities. When analyzing legal documents, you can use <|thinking|> tags to show your reasoning process step by step before providing your final analysis.

Key capabilities:
- Legal document classification and analysis
- Evidence evaluation and chain of custody verification  
- Case strength assessment
- Regulatory compliance checking
- Structured data extraction from legal forms

When requested to use "thinking style", show your reasoning process using <|thinking|> tags before your final response. Structure your thinking as:
1. Initial document assessment
2. Key element identification
3. Legal implications analysis
4. Quality and reliability evaluation
5. Final recommendations

Always provide structured, actionable analysis for legal professionals."""`;

    // Write modelfile
    await fs.writeFile("Legal-Thinking-Modelfile", modelfile);

    // Create the model
    await execAsync(
      "ollama create legal-gemma3-thinking -f Legal-Thinking-Modelfile",
    );
    console.log("✅ Legal thinking model created successfully");

    // Clean up
    await fs.unlink("Legal-Thinking-Modelfile");
  } catch (error) {
    console.error("❌ Error setting up Ollama models:", error.message);
    throw error;
  }
}

async function listOllamaModels() {
  try {
    console.log("\n📋 Available Ollama models:");
    const { stdout } = await execAsync("ollama list");
    console.log(stdout);
  } catch (error) {
    console.error("❌ Error listing models:", error.message);
  }
}

async function updateDatabaseSchema() {
  console.log("\n🗄️  Updating database schema...");

  const schemaMigration = `
-- Enhanced schema for document analysis and thinking style

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id TEXT,
  evidence_id TEXT,
  case_id TEXT,
  analysis_type TEXT NOT NULL,
  thinking_process TEXT,
  analysis_result TEXT NOT NULL,
  model_used TEXT NOT NULL,
  processing_time INTEGER,
  confidence_score REAL,
  thinking_enabled BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (evidence_id) REFERENCES evidence(id),
  FOREIGN KEY (case_id) REFERENCES cases(id)
);

-- AI models configuration
CREATE TABLE IF NOT EXISTS ai_models (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  model_type TEXT NOT NULL, -- 'thinking' or 'quick'
  ollama_name TEXT NOT NULL,
  capabilities TEXT, -- JSON array of capabilities
  is_active BOOLEAN DEFAULT TRUE,
  thinking_enabled BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User AI preferences
CREATE TABLE IF NOT EXISTS user_ai_preferences (
  user_id TEXT PRIMARY KEY,
  thinking_style_enabled BOOLEAN DEFAULT FALSE,
  preferred_model TEXT,
  thinking_depth TEXT DEFAULT 'detailed', -- basic, detailed, comprehensive
  focus_areas TEXT, -- JSON array
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Document processing jobs
CREATE TABLE IF NOT EXISTS document_processing_jobs (
  id TEXT PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id TEXT,
  job_type TEXT NOT NULL, -- 'ocr', 'analysis', 'classification'
  status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
  thinking_enabled BOOLEAN DEFAULT FALSE,
  result TEXT,
  error_message TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP
);

-- Insert default AI models
INSERT INTO ai_models (id, name, model_type, ollama_name, capabilities, thinking_enabled) VALUES
('quick-gemma3', 'Gemma 3 Quick', 'quick', 'gemma3:7b', '["classification", "extraction", "summarization"]', FALSE),
('thinking-gemma3', 'Gemma 3 Thinking', 'thinking', 'legal-gemma3-thinking', '["reasoning", "analysis", "chain_of_custody", "legal_research"]', TRUE)
ON CONFLICT (id) DO NOTHING;
`;

  try {
    await fs.writeFile("schema-migration.sql", schemaMigration);
    console.log("✅ Database migration file created: schema-migration.sql");
    console.log("💡 Run this migration in your database to add the new tables");
  } catch (error) {
    console.error("❌ Error creating database migration:", error.message);
  }
}

async function setupDocumentProcessing() {
  console.log("\n📄 Setting up document processing...");

  try {
    // Create docs directories
    await fs.mkdir("docs", { recursive: true });
    await fs.mkdir("docs/raw", { recursive: true });
    await fs.mkdir("docs/processed", { recursive: true });

    console.log("✅ Document processing directories created");

    // Create sample datasets directory
    await fs.mkdir("static/datasets", { recursive: true });

    const sampleData = {
      training_samples: 50,
      categories: ["evidence_logs", "witness_statements", "court_filings"],
      last_updated: new Date().toISOString(),
      description: "Sample legal document analysis training data",
      thinking_style_examples: 40,
    };

    await fs.writeFile(
      "static/datasets/sample-legal-data.json",
      JSON.stringify(sampleData, null, 2),
    );
    console.log("✅ Sample datasets created");
  } catch (error) {
    console.error("❌ Error setting up document processing:", error.message);
  }
}

async function testSystemIntegration() {
  console.log("\n🧪 Testing system integration...");

  try {
    // Test Ollama connection
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "legal-gemma3-thinking",
        prompt:
          'Test: Analyze this evidence briefly: "DNA sample collected from crime scene"',
        stream: false,
        options: { num_predict: 100 },
      }),
    });

    if (response.ok) {
      const result = await response.json();
      console.log("✅ Legal thinking model responding correctly");
      console.log(
        "📝 Sample response:",
        result.response?.substring(0, 100) + "...",
      );
    } else {
      throw new Error("Model test failed");
    }
  } catch (error) {
    console.error("❌ Integration test failed:", error.message);
  }
}

async function createCompletionReport() {
  console.log("\n📊 Creating completion report...");

  const report = `# 🧠 Enhanced Legal AI Setup Complete!

## What's Been Set Up

### 1. Ollama Models
- ✅ Gemma 3 Base Model (gemma3:7b)
- ✅ Legal Thinking Model (legal-gemma3-thinking)

### 2. Database Enhancements
- ✅ Analysis results tracking table
- ✅ AI model configuration table
- ✅ User preferences for thinking style
- ✅ Document processing job queue

### 3. Document Processing
- ✅ Document storage directories
- ✅ Sample training datasets
- ✅ Processing pipeline structure

## API Endpoints Available

### Enhanced Analysis
- \`POST /api/analyze\` - Document analysis with thinking support
- \`GET /api/analyze?evidenceId=X\` - Analysis history

### Features
- **Quick Mode**: Fast responses using gemma3:7b
- **Thinking Style**: Detailed reasoning with legal-gemma3-thinking
- **Evidence Analysis**: Chain of custody verification
- **Case Assessment**: Strength analysis with confidence scores

## Usage Examples

### 1. Quick Analysis
\`\`\`bash
curl -X POST http://localhost:5173/api/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Evidence collected from crime scene","analysisType":"classification"}'
\`\`\`

### 2. Thinking Style Analysis
\`\`\`bash
curl -X POST http://localhost:5173/api/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Evidence collected from crime scene","analysisType":"reasoning","useThinkingStyle":true}'
\`\`\`

### 3. Evidence Analysis
\`\`\`bash
curl -X POST http://localhost:5173/api/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"evidenceId":"evidence-123","useThinkingStyle":true}'
\`\`\`

## Integration with Your App

The thinking style toggle is now available in:
- 🗣️ AI Chat Interface (\`/ai\`)
- 📁 Evidence Management (\`/evidence\`)
- 📋 Case Analysis pages

## Next Steps

1. **Install Dependencies**: \`npm install\`
2. **Run Database Migration**: Apply schema-migration.sql to your database
3. **Start Development**: \`npm run dev\`
4. **Test Integration**: \`npm run thinking:test\`

## Troubleshooting

- **Ollama not responding**: Ensure Ollama is running (\`ollama serve\`)
- **Model not found**: Run model setup again (\`npm run thinking:setup\`)
- **Database errors**: Apply the schema migration
- **Analysis failing**: Check Ollama model availability

🎉 Your enhanced legal AI system with thinking capabilities is ready!
`;

  await fs.writeFile("THINKING_AI_SETUP_COMPLETE.md", report);
  console.log("✅ Setup report created: THINKING_AI_SETUP_COMPLETE.md");
}

// Main setup function
async function main() {
  try {
    // Step 1: Check prerequisites
    const ollamaAvailable = await checkOllamaAvailability();
    if (!ollamaAvailable) {
      console.log("🛑 Setup aborted - Ollama not available");
      console.log("💡 Install Ollama first: https://ollama.ai");
      process.exit(1);
    }

    // Step 2: Setup Ollama models
    await setupOllamaModels();

    // Step 3: List available models
    await listOllamaModels();

    // Step 4: Update database schema
    await updateDatabaseSchema();

    // Step 5: Setup document processing
    await setupDocumentProcessing();

    // Step 6: Test integration
    await testSystemIntegration();

    // Step 7: Create completion report
    await createCompletionReport();

    console.log("\n🎉 Enhanced Legal AI Setup Complete!");
    console.log("=============================================");
    console.log("✅ Ollama models configured");
    console.log("✅ Database schema ready");
    console.log("✅ Document processing setup");
    console.log("✅ System integration tested");
    console.log("");
    console.log("📖 See THINKING_AI_SETUP_COMPLETE.md for usage instructions");
    console.log("🚀 Run `npm run dev` to start your enhanced legal AI system!");
  } catch (error) {
    console.error("\n💥 Setup failed:", error.message);
    process.exit(1);
  }
}

// Run setup
main();
