/**
 * Best Practices Generation Service
 * Agentic workflow for analyzing codebase and generating actionable best practices
 */

import { aiService } from "./ai-service.js";
import { redisVectorService } from "./redis-vector-service.js";
import { documentIngestionService } from "./document-ingestion-service.js";
import fs from "fs/promises";
import path from "path";
import { glob } from "glob";

export interface BestPractice {
  id: string;
  title: string;
  category:
    | "security"
    | "performance"
    | "maintainability"
    | "testing"
    | "architecture"
    | "accessibility";
  priority: "high" | "medium" | "low";
  description: string;
  rationale: string;
  examples: {
    good?: string;
    bad?: string;
  };
  resources: string[];
  actionable_steps: string[];
  estimated_effort: "low" | "medium" | "high";
  impact: "low" | "medium" | "high";
}

export interface CodebaseAnalysis {
  project_type: string;
  technologies: string[];
  file_count: number;
  lines_of_code: number;
  patterns_detected: string[];
  potential_issues: string[];
  strengths: string[];
  architecture_style: string;
}

export interface BestPracticesReport {
  id: string;
  timestamp: string;
  codebase_analysis: CodebaseAnalysis;
  best_practices: BestPractice[];
  summary: {
    total_recommendations: number;
    high_priority_count: number;
    estimated_total_effort: string;
    quick_wins: BestPractice[];
  };
}

export class BestPracticesService {
  private promptTemplate = `
You are an expert software architect and code quality consultant. Analyze the provided codebase information and generate specific, actionable best practices.

CODEBASE ANALYSIS:
{codebase_analysis}

DETECTED PATTERNS:
{patterns}

POTENTIAL ISSUES:
{issues}

Generate 5-10 high-quality best practices following this JSON schema:
{
  "best_practices": [
    {
      "title": "Clear, specific title",
      "category": "security|performance|maintainability|testing|architecture|accessibility",
      "priority": "high|medium|low",
      "description": "Detailed description of the practice",
      "rationale": "Why this practice is important for this specific codebase",
      "examples": {
        "good": "Code example showing correct implementation",
        "bad": "Code example showing what to avoid"
      },
      "resources": ["URL or reference to learn more"],
      "actionable_steps": ["Step 1", "Step 2", "Step 3"],
      "estimated_effort": "low|medium|high",
      "impact": "low|medium|high"
    }
  ]
}

Focus on:
1. Practices specific to the detected technologies
2. Issues that can be addressed quickly (quick wins)
3. High-impact architectural improvements
4. Security vulnerabilities specific to the stack
5. Performance optimizations relevant to the codebase

Make recommendations specific, actionable, and prioritized by impact vs effort.
`;

  constructor() {}

  /**
   * Generate best practices for a given project directory
   */
  async generateBestPractices(
    projectPath: string
  ): Promise<BestPracticesReport> {
    console.log(`🔍 Analyzing codebase at: ${projectPath}`);

    // Step 1: Analyze codebase structure and patterns
    const codebaseAnalysis = await this.analyzeCodebase(projectPath);

    // Step 2: Detect patterns and potential issues
    const patterns = await this.detectPatterns(projectPath, codebaseAnalysis);
    const issues = await this.detectIssues(projectPath, codebaseAnalysis);

    // Step 3: Generate contextual best practices using AI
    const bestPractices = await this.generateContextualPractices(
      codebaseAnalysis,
      patterns,
      issues
    );

    // Step 4: Prioritize and enhance recommendations
    const prioritizedPractices = this.prioritizePractices(bestPractices);

    // Step 5: Create comprehensive report
    const report: BestPracticesReport = {
      id: this.generateReportId(),
      timestamp: new Date().toISOString(),
      codebase_analysis: codebaseAnalysis,
      best_practices: prioritizedPractices,
      summary: this.generateSummary(prioritizedPractices),
    };

    // Step 6: Store in vector database for future reference
    await this.storeReport(report);

    console.log(`✅ Generated ${bestPractices.length} best practices`);
    return report;
  }

  /**
   * Analyze codebase structure and extract metadata
   */
  private async analyzeCodebase(
    projectPath: string
  ): Promise<CodebaseAnalysis> {
    const files = await glob("**/*.{ts,js,svelte,tsx,jsx,py,css,html}", {
      cwd: projectPath,
      ignore: ["**/node_modules/**", "**/dist/**", "**/build/**", "**/.git/**"],
    });

    let totalLines = 0;
    const technologies = new Set<string>();
    const patterns = new Set<string>();

    // Analyze files
    for (const file of files.slice(0, 100)) {
      // Limit analysis for performance
      try {
        const fullPath = path.join(projectPath, file);
        const content = await fs.readFile(fullPath, "utf-8");
        const lines = content.split("\n").length;
        totalLines += lines;

        // Detect technologies
        if (file.endsWith(".svelte")) technologies.add("Svelte");
        if (file.endsWith(".tsx") || file.endsWith(".jsx"))
          technologies.add("React");
        if (file.endsWith(".ts")) technologies.add("TypeScript");
        if (file.endsWith(".py")) technologies.add("Python");
        if (content.includes("docker")) technologies.add("Docker");
        if (content.includes("redis")) technologies.add("Redis");
        if (content.includes("postgres")) technologies.add("PostgreSQL");

        // Detect patterns
        if (content.includes("TODO") || content.includes("FIXME"))
          patterns.add("TODOs_present");
        if (content.includes("console.log")) patterns.add("debug_statements");
        if (content.includes("any")) patterns.add("loose_typing");
        if (content.includes("fetch(")) patterns.add("api_calls");
        if (content.includes("await")) patterns.add("async_patterns");
      } catch (error) {
        console.warn(`Failed to analyze file ${file}:`, error);
      }
    }

    // Detect project type and architecture
    const packageJsonPath = path.join(projectPath, "package.json");
    let projectType = "unknown";
    let architectureStyle = "unknown";

    try {
      const packageJson = JSON.parse(
        await fs.readFile(packageJsonPath, "utf-8")
      );
      if (packageJson.dependencies?.["@sveltejs/kit"]) {
        projectType = "SvelteKit";
        architectureStyle = "SSR_Framework";
      }
      if (packageJson.dependencies?.react) {
        projectType = "React";
        architectureStyle = "SPA";
      }
      if (packageJson.dependencies?.express) {
        architectureStyle = "REST_API";
      }
    } catch (error) {
      console.warn("Could not read package.json:", error);
    }

    return {
      project_type: projectType,
      technologies: Array.from(technologies),
      file_count: files.length,
      lines_of_code: totalLines,
      patterns_detected: Array.from(patterns),
      potential_issues: [], // Will be filled by detectIssues
      strengths: [], // Will be filled by detectPatterns
      architecture_style: architectureStyle,
    };
  }

  /**
   * Detect positive patterns and strengths
   */
  private async detectPatterns(
    projectPath: string,
    analysis: CodebaseAnalysis
  ): Promise<string[]> {
    const patterns: string[] = [];

    // Check for good practices
    if (analysis.technologies.includes("TypeScript")) {
      patterns.push("Strong typing with TypeScript");
    }

    if (analysis.patterns_detected.includes("async_patterns")) {
      patterns.push("Modern async/await patterns");
    }

    // Check for testing
    const testFiles = await glob("**/*.{test,spec}.{ts,js}", {
      cwd: projectPath,
      ignore: ["**/node_modules/**"],
    });

    if (testFiles.length > 0) {
      patterns.push(`Testing setup with ${testFiles.length} test files`);
    }

    // Check for documentation
    try {
      await fs.access(path.join(projectPath, "README.md"));
      patterns.push("Documentation with README");
    } catch (error) {
      // README doesn't exist
    }

    return patterns;
  }

  /**
   * Detect potential issues and anti-patterns
   */
  private async detectIssues(
    projectPath: string,
    analysis: CodebaseAnalysis
  ): Promise<string[]> {
    const issues: string[] = [];

    if (analysis.patterns_detected.includes("TODOs_present")) {
      issues.push("Multiple TODO/FIXME comments indicate incomplete features");
    }

    if (analysis.patterns_detected.includes("debug_statements")) {
      issues.push("Debug console.log statements found in codebase");
    }

    if (analysis.patterns_detected.includes("loose_typing")) {
      issues.push('Usage of "any" type reduces type safety');
    }

    if (
      analysis.file_count > 200 &&
      !analysis.technologies.includes("TypeScript")
    ) {
      issues.push("Large codebase without strong typing system");
    }

    // Check for security issues
    try {
      const envFile = await fs.readFile(
        path.join(projectPath, ".env"),
        "utf-8"
      );
      if (envFile.includes("password") || envFile.includes("secret")) {
        issues.push("Potential secrets in .env file");
      }
    } catch (error) {
      // .env file doesn't exist or isn't readable
    }

    return issues;
  }

  /**
   * Generate contextual best practices using AI
   */
  private async generateContextualPractices(
    analysis: CodebaseAnalysis,
    patterns: string[],
    issues: string[]
  ): Promise<BestPractice[]> {
    const prompt = this.promptTemplate
      .replace("{codebase_analysis}", JSON.stringify(analysis, null, 2))
      .replace("{patterns}", patterns.join("\n- "))
      .replace("{issues}", issues.join("\n- "));

    try {
      const response = await aiService.generateText(prompt, {
        provider: "auto",
        temperature: 0, // Deterministic for consistent results
        maxTokens: 4000,
      });

      const result = JSON.parse(response);
      return result.best_practices.map((practice: any, index: number) => ({
        id: `bp_${Date.now()}_${index}`,
        ...practice,
      }));
    } catch (error) {
      console.error("Failed to generate AI practices:", error);
      return this.getFallbackPractices(analysis);
    }
  }

  /**
   * Prioritize practices by impact vs effort
   */
  private prioritizePractices(practices: BestPractice[]): BestPractice[] {
    return practices.sort((a, b) => {
      // Priority order: high > medium > low
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const impactOrder = { high: 3, medium: 2, low: 1 };
      const effortOrder = { low: 3, medium: 2, high: 1 }; // Low effort = higher score

      const scoreA =
        priorityOrder[a.priority] +
        impactOrder[a.impact] +
        effortOrder[a.estimated_effort];
      const scoreB =
        priorityOrder[b.priority] +
        impactOrder[b.impact] +
        effortOrder[b.estimated_effort];

      return scoreB - scoreA;
    });
  }

  /**
   * Generate report summary
   */
  private generateSummary(practices: BestPractice[]) {
    const highPriorityCount = practices.filter(
      (p) => p.priority === "high"
    ).length;
    const quickWins = practices
      .filter(
        (p) =>
          p.estimated_effort === "low" &&
          (p.impact === "medium" || p.impact === "high")
      )
      .slice(0, 3);

    return {
      total_recommendations: practices.length,
      high_priority_count: highPriorityCount,
      estimated_total_effort: this.calculateTotalEffort(practices),
      quick_wins: quickWins,
    };
  }

  /**
   * Calculate total effort estimate
   */
  private calculateTotalEffort(practices: BestPractice[]): string {
    const effortCounts = practices.reduce(
      (acc, p) => {
        acc[p.estimated_effort] = (acc[p.estimated_effort] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    const totalPoints =
      (effortCounts.low || 0) * 1 +
      (effortCounts.medium || 0) * 3 +
      (effortCounts.high || 0) * 8;

    if (totalPoints <= 5) return "low";
    if (totalPoints <= 15) return "medium";
    return "high";
  }

  /**
   * Store report in vector database
   */
  private async storeReport(report: BestPracticesReport): Promise<void> {
    try {
      // Generate embedding for the report content
      const content = `${report.codebase_analysis.project_type} best practices: ${report.best_practices
        .map((p) => p.title)
        .join(", ")}`;

      const embedding = await aiService.generateEmbedding(content, {
        provider: "auto",
      });

      await redisVectorService.storeDocument({
        id: report.id,
        embedding,
        metadata: {
          type: "best_practices_report",
          project_type: report.codebase_analysis.project_type,
          technologies: report.codebase_analysis.technologies,
          timestamp: report.timestamp,
          practices_count: report.best_practices.length,
        },
        content: JSON.stringify(report),
        ttl: 604800, // 1 week
      });
    } catch (error) {
      console.error("Failed to store best practices report:", error);
    }
  }

  /**
   * Fallback practices when AI generation fails
   */
  private getFallbackPractices(analysis: CodebaseAnalysis): BestPractice[] {
    const fallbacks: BestPractice[] = [];

    if (analysis.technologies.includes("TypeScript")) {
      fallbacks.push({
        id: "fallback_1",
        title: "Strengthen TypeScript Configuration",
        category: "maintainability",
        priority: "medium",
        description:
          "Enable strict TypeScript compiler options for better type safety",
        rationale:
          "Strict typing prevents runtime errors and improves code maintainability",
        examples: {
          good: '{ "strict": true, "noImplicitReturns": true }',
          bad: '{ "strict": false }',
        },
        resources: ["https://www.typescriptlang.org/tsconfig#strict"],
        actionable_steps: [
          "Enable strict mode in tsconfig.json",
          "Fix any type errors that surface",
          "Add explicit return types to functions",
        ],
        estimated_effort: "medium",
        impact: "high",
      });
    }

    return fallbacks;
  }

  /**
   * Generate unique report ID
   */
  private generateReportId(): string {
    return `bp_report_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get stored best practices reports
   */
  async getStoredReports(limit: number = 10): Promise<BestPracticesReport[]> {
    try {
      // Search for best practices reports in vector DB
      const dummyEmbedding = new Array(384).fill(0);
      const results = await redisVectorService.searchSimilar(dummyEmbedding, {
        topK: limit,
        threshold: 0,
        filter: { type: "best_practices_report" },
      });

      return results.map((result) => JSON.parse(result.content));
    } catch (error) {
      console.error("Failed to retrieve stored reports:", error);
      return [];
    }
  }
}

// Export singleton instance
export const bestPracticesService = new BestPracticesService();

// Export for use in other services
export default bestPracticesService;
