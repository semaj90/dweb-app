#!/usr/bin/env node

/**
 * Document Fetching Script for Legal AI (Cross-platform)
 * Downloads curated legal and technical documents for AI training
 */

import { promises as fs } from "fs";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

console.log("📥 Fetching documents for Legal AI...");
console.log("====================================");

// Document sources for AI training
const documentSources = [
  // Technical documentation
  {
    url: "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/index.md",
    category: "technical",
    description: "JavaScript Guide",
  },
  {
    url: "https://raw.githubusercontent.com/sveltejs/kit/main/README.md",
    category: "technical",
    description: "SvelteKit Documentation",
  },

  // Legal resources (public domain)
  {
    url: "https://www.law.cornell.edu/constitution/overview",
    category: "legal",
    description: "Constitutional Law Overview",
  },
  {
    url: "https://www.uscourts.gov/about-federal-courts/educational-resources/about-educational-outreach/activity-resources/what",
    category: "legal",
    description: "Federal Courts Educational Resources",
  },

  // Sample legal documents (public domain)
  {
    url: "https://www.supremecourt.gov/opinions/boundvolumes/",
    category: "legal",
    description: "Supreme Court Opinions",
  },
];

// Additional local sample documents to create
const sampleDocuments = [
  {
    filename: "evidence_collection_guidelines.md",
    content: `# Evidence Collection Guidelines

## Chain of Custody Requirements

### Initial Collection
1. **Documentation**: All evidence must be properly documented at the time of collection
2. **Labeling**: Each piece of evidence requires unique identification
3. **Photography**: Photograph evidence in place before collection
4. **Witness**: Have a witness present during collection when possible

### Transfer Procedures
- Maintain continuous custody records
- Document all transfers between personnel
- Include date, time, and personnel involved
- Ensure proper storage conditions

### Legal Standards
Evidence collection must comply with Fourth Amendment requirements:
- Valid search warrant (with exceptions)
- Probable cause established
- Miranda rights when applicable
- Due process protections maintained

## Best Practices
- Use proper collection tools and containers
- Avoid contamination
- Maintain detailed logs
- Follow departmental protocols
- Ensure admissibility standards are met
`,
    category: "legal",
  },
  {
    filename: "witness_statement_template.md",
    content: `# Witness Statement Template

## Case Information
- **Case Number**: ___________
- **Date**: ___________
- **Location**: ___________
- **Investigating Officer**: ___________

## Witness Information
- **Name**: ___________
- **Address**: ___________
- **Phone**: ___________
- **Date of Birth**: ___________

## Statement
**I, [Witness Name], hereby state the following:**

### What I Observed
[Detailed description of events witnessed]

### When It Occurred
- **Date**: ___________
- **Time**: ___________
- **Duration**: ___________

### Where It Occurred
[Specific location details]

### Additional Details
[Any other relevant information]

## Verification
I certify that the above statement is true and accurate to the best of my knowledge.

**Witness Signature**: ___________ **Date**: ___________
**Officer Signature**: ___________ **Date**: ___________

---
*This statement was given voluntarily and may be used in legal proceedings.*
`,
    category: "legal",
  },
  {
    filename: "case_analysis_framework.md",
    content: `# Legal Case Analysis Framework

## Case Strength Assessment

### Evidence Evaluation Criteria
1. **Reliability**: How dependable is the evidence?
2. **Authenticity**: Can the evidence be verified?
3. **Relevance**: Does it directly relate to the case?
4. **Admissibility**: Will it be allowed in court?

### Strength Indicators
- **Strong Case**: Multiple corroborating evidence sources
- **Moderate Case**: Some evidence gaps but overall solid
- **Weak Case**: Insufficient or unreliable evidence

### Risk Factors
- Chain of custody issues
- Constitutional violations
- Witness credibility problems
- Procedural errors

## Legal Analysis Process

### Step 1: Fact Gathering
- Collect all available evidence
- Interview witnesses
- Review documentation
- Analyze physical evidence

### Step 2: Legal Research
- Identify applicable laws
- Research relevant precedents
- Review procedural requirements
- Assess constitutional issues

### Step 3: Strategy Development
- Evaluate prosecution prospects
- Identify potential defenses
- Plan evidence presentation
- Consider plea negotiations

### Step 4: Risk Assessment
- Evaluate likelihood of conviction
- Assess potential sentences
- Consider appeal prospects
- Review resource requirements

## Decision Framework
Use this framework to make informed decisions about case prosecution and strategy.
`,
    category: "legal",
  },
];

async function createDirectories() {
  console.log("📁 Creating directory structure...");

  try {
    await fs.mkdir("docs", { recursive: true });
    await fs.mkdir("docs/raw", { recursive: true });
    await fs.mkdir("docs/processed", { recursive: true });
    await fs.mkdir("docs/samples", { recursive: true });

    console.log("✅ Directory structure created");
  } catch (error) {
    console.error("❌ Error creating directories:", error.message);
    throw error;
  }
}

async function downloadDocument(source) {
  const filename = generateFilename(source.url, source.description);
  const filepath = path.join("docs", "raw", filename);

  try {
    console.log(`📥 Downloading: ${source.description}`);

    // Try using curl (cross-platform)
    try {
      await execAsync(`curl -sL "${source.url}" -o "${filepath}"`, {
        timeout: 30000,
      });

      // Check if file was created and has content
      const stats = await fs.stat(filepath);
      if (stats.size > 0) {
        console.log(`✅ Downloaded: ${filename} (${stats.size} bytes)`);
        return true;
      } else {
        throw new Error("Downloaded file is empty");
      }
    } catch (curlError) {
      // Fallback to Node.js fetch
      console.log(`⚠️  Curl failed, trying fetch...`);
      return await downloadWithFetch(source.url, filepath);
    }
  } catch (error) {
    console.log(
      `❌ Failed to download ${source.description}: ${error.message}`,
    );
    return false;
  }
}

async function downloadWithFetch(url, filepath) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const content = await response.text();
    await fs.writeFile(filepath, content, "utf-8");

    console.log(
      `✅ Downloaded with fetch: ${path.basename(filepath)} (${content.length} chars)`,
    );
    return true;
  } catch (error) {
    console.log(`❌ Fetch failed: ${error.message}`);
    return false;
  }
}

async function createSampleDocuments() {
  console.log("\n📝 Creating sample legal documents...");

  let createdCount = 0;

  for (const doc of sampleDocuments) {
    try {
      const filepath = path.join("docs", "samples", doc.filename);
      await fs.writeFile(filepath, doc.content, "utf-8");
      console.log(`✅ Created: ${doc.filename}`);
      createdCount++;
    } catch (error) {
      console.error(`❌ Error creating ${doc.filename}:`, error.message);
    }
  }

  console.log(`✅ Created ${createdCount} sample documents`);
  return createdCount;
}

function generateFilename(url, description) {
  // Create a safe filename from URL and description
  const urlPart = url
    .replace(/^https?:\/\//, "")
    .replace(/[^a-zA-Z0-9]/g, "_")
    .substring(0, 50);

  const descPart = description
    .replace(/[^a-zA-Z0-9\s]/g, "")
    .replace(/\s+/g, "_")
    .toLowerCase()
    .substring(0, 30);

  return `${descPart}_${urlPart}.html`;
}

async function createDocumentIndex() {
  console.log("\n📊 Creating document index...");

  try {
    const rawFiles = await fs.readdir("docs/raw");
    const sampleFiles = await fs.readdir("docs/samples");

    const index = {
      createdAt: new Date().toISOString(),
      totalDocuments: rawFiles.length + sampleFiles.length,
      categories: {
        downloaded: rawFiles.length,
        samples: sampleFiles.length,
      },
      rawDocuments: rawFiles.map((file) => ({
        filename: file,
        path: `docs/raw/${file}`,
        type: "downloaded",
      })),
      sampleDocuments: sampleFiles.map((file) => ({
        filename: file,
        path: `docs/samples/${file}`,
        type: "sample",
      })),
      processingInstructions: {
        nextStep:
          "Run `npm run docs:process` to process these documents for AI analysis",
        outputDirectory: "docs/processed",
        aiReady: false,
      },
    };

    await fs.writeFile(
      "docs/document-index.json",
      JSON.stringify(index, null, 2),
    );
    console.log("✅ Document index created: docs/document-index.json");

    return index;
  } catch (error) {
    console.error("❌ Error creating document index:", error.message);
    return null;
  }
}

async function checkSystemRequirements() {
  console.log("🔍 Checking system requirements...");

  // Check if curl is available
  try {
    await execAsync("curl --version");
    console.log("✅ curl is available");
  } catch (error) {
    console.log("⚠️  curl not available, will use Node.js fetch as fallback");
  }

  // Check Node.js version
  const nodeVersion = process.version;
  console.log(`✅ Node.js version: ${nodeVersion}`);

  // Check if we have internet connectivity
  try {
    const response = await fetch("https://www.google.com", {
      method: "HEAD",
      timeout: 5000,
    });
    if (response.ok) {
      console.log("✅ Internet connectivity confirmed");
    }
  } catch (error) {
    console.log("⚠️  Internet connectivity check failed");
    console.log("💡 Some downloads may fail without internet access");
  }
}

async function main() {
  try {
    // Check system requirements
    await checkSystemRequirements();

    // Create directory structure
    await createDirectories();

    // Download documents
    console.log("\n📥 Downloading external documents...");
    let downloadCount = 0;

    for (const source of documentSources) {
      const success = await downloadDocument(source);
      if (success) downloadCount++;
    }

    console.log(
      `✅ Downloaded ${downloadCount}/${documentSources.length} external documents`,
    );

    // Create sample documents
    const sampleCount = await createSampleDocuments();

    // Create document index
    const index = await createDocumentIndex();

    // Final summary
    console.log("\n📋 Document Fetching Summary");
    console.log("============================");
    console.log(`📥 External documents downloaded: ${downloadCount}`);
    console.log(`📝 Sample documents created: ${sampleCount}`);
    console.log(
      `📊 Total documents available: ${index ? index.totalDocuments : downloadCount + sampleCount}`,
    );
    console.log("📁 Documents stored in: docs/raw and docs/samples");

    console.log("\n🎉 Document fetching complete!");
    console.log("💡 Next steps:");
    console.log("   1. Run `npm run docs:process` to process documents for AI");
    console.log("   2. Documents will be available for legal AI analysis");
    console.log("   3. Use processed documents with /api/analyze endpoint");
  } catch (error) {
    console.error("\n💥 Document fetching failed:", error.message);
    process.exit(1);
  }
}

// Add error handling for common issues
process.on("unhandledRejection", (reason, promise) => {
  console.error("💥 Unhandled rejection at:", promise, "reason:", reason);
  process.exit(1);
});

// Run the script
main();
