#!/usr/bin/env node

/**
 * Multi-Agent Evidence Analysis Pipeline
 * Coordinates 4 specialized agents to analyze legal evidence
 */

import { spawn } from 'child_process';
import { readFile, writeFile, mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class EvidenceAnalysisPipeline {
  constructor(caseId, evidenceFile) {
    this.caseId = caseId;
    this.evidenceFile = evidenceFile;
    this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.tempDir = join(process.cwd(), 'temp', `analysis_${caseId}_${this.timestamp}`);
    this.claudeAgentsDir = join(process.cwd(), '.claude', 'agents');
  }

  async run() {
    try {
      console.log('🚀 Starting Multi-Agent Evidence Analysis Pipeline');
      console.log(`📁 Case ID: ${this.caseId}`);
      console.log(`📄 Evidence: ${this.evidenceFile}`);
      console.log(`🗂️  Working Directory: ${this.tempDir}`);

      // Create temp directory
      await mkdir(this.tempDir, { recursive: true });

      // Step 1: Evidence Analyzer Agent
      console.log('\\n=== Step 1: Analyzing Evidence ===');
      const evidenceAnalysis = await this.runAgent('evidence-analyzer', this.evidenceFile);
      await this.saveOutput('evidence_analysis.json', evidenceAnalysis);

      // Step 2: Person Extractor Agent  
      console.log('\\n=== Step 2: Extracting Persons of Interest ===');
      const personsExtracted = await this.runAgent('person-extractor', evidenceAnalysis);
      await this.saveOutput('persons_extracted.json', personsExtracted);

      // Step 3: Relationship Mapper Agent
      console.log('\\n=== Step 3: Mapping Relationships to Neo4j ===');
      const mapperInput = {
        caseId: this.caseId,
        evidenceFile: this.evidenceFile,
        personsData: JSON.parse(personsExtracted)
      };
      const neo4jUpdates = await this.runAgent('relationship-mapper', JSON.stringify(mapperInput));
      await this.saveOutput('neo4j_updates.json', neo4jUpdates);

      // Step 4: Case Synthesizer Agent
      console.log('\\n=== Step 4: Synthesizing Case Analysis ===');
      const synthesisInput = {
        caseId: this.caseId,
        evidenceAnalysis: JSON.parse(evidenceAnalysis),
        personsData: JSON.parse(personsExtracted),
        neo4jUpdates: JSON.parse(neo4jUpdates)
      };
      const caseSynthesis = await this.runAgent('case-synthesizer', JSON.stringify(synthesisInput));
      await this.saveOutput('case_synthesis.json', caseSynthesis);

      console.log('\\n✅ Multi-agent analysis pipeline completed successfully!');
      console.log('\\n📊 Output files:');
      console.log(`   - Evidence Analysis: ${join(this.tempDir, 'evidence_analysis.json')}`);
      console.log(`   - Persons Extracted: ${join(this.tempDir, 'persons_extracted.json')}`);
      console.log(`   - Neo4j Updates: ${join(this.tempDir, 'neo4j_updates.json')}`);
      console.log(`   - Case Synthesis: ${join(this.tempDir, 'case_synthesis.json')}`);

      return {
        success: true,
        tempDir: this.tempDir,
        outputs: {
          evidence_analysis: JSON.parse(evidenceAnalysis),
          persons_extracted: JSON.parse(personsExtracted),
          neo4j_updates: JSON.parse(neo4jUpdates),
          case_synthesis: JSON.parse(caseSynthesis)
        }
      };

    } catch (error) {
      console.error('❌ Analysis pipeline failed:', error.message);
      return {
        success: false,
        error: error.message,
        tempDir: this.tempDir
      };
    }
  }

  async runAgent(agentName, input) {
    const agentFile = join(this.claudeAgentsDir, `${agentName}.md`);
    const inputFile = join(this.tempDir, `${agentName}_input.txt`);

    try {
      // Write input to temporary file
      if (typeof input === 'object') {
        await writeFile(inputFile, JSON.stringify(input, null, 2));
      } else {
        await writeFile(inputFile, input);
      }

      // Run Claude with the agent
      console.log(`🤖 Running ${agentName} agent...`);
      const result = await this.executeClaudeCommand(agentFile, inputFile);
      
      console.log(`✅ ${agentName} completed`);
      return result;

    } catch (error) {
      console.error(`❌ ${agentName} failed:`, error.message);
      throw error;
    }
  }

  async executeClaudeCommand(agentFile, inputFile) {
    return new Promise((resolve, reject) => {
      // Use Claude CLI command
      const claude = spawn('claude', ['-f', agentFile, '--input-file', inputFile], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      claude.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      claude.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      claude.on('close', (code) => {
        if (code === 0) {
          resolve(stdout.trim());
        } else {
          reject(new Error(`Claude command failed with code ${code}: ${stderr}`));
        }
      });

      claude.on('error', (error) => {
        reject(new Error(`Failed to execute Claude: ${error.message}`));
      });
    });
  }

  async saveOutput(filename, content) {
    const filepath = join(this.tempDir, filename);
    await writeFile(filepath, content);
    console.log(`💾 Saved ${filename}`);
  }
}

// Command line interface
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.error('Usage: node analyze-evidence.js <case_id> <evidence_file_path>');
    process.exit(1);
  }

  const [caseId, evidenceFile] = args;

  // Validate evidence file exists
  try {
    await readFile(evidenceFile);
  } catch (error) {
    console.error(`Error: Evidence file not found: ${evidenceFile}`);
    process.exit(1);
  }

  // Run the pipeline
  const pipeline = new EvidenceAnalysisPipeline(caseId, evidenceFile);
  const result = await pipeline.run();

  if (!result.success) {
    process.exit(1);
  }
}

// Export for use as module
export { EvidenceAnalysisPipeline };

// Run if called directly
if (import.meta.url === `file://${__filename}`) {
  main().catch(console.error);
}