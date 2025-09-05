import { Pool } from "pg";
import { drizzle } from "drizzle-orm/node-postgres";
import * as schema from "../src/lib/server/db/unified-schema.ts";
import dotenv from "dotenv";

// Load environment variables
dotenv.config({ path: ".env.development" });

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

const db = drizzle(pool, { schema });

async function seedSampleData() {
  console.log("🌱 Seeding sample cases, evidence, and reports...");

  try {
    // Get the admin user
    const adminUser = await db.query.users.findFirst({
      where: (users, { eq }) => eq(users.email, "admin@prosecutor.com"),
    });

    if (!adminUser) {
      console.error(
        "❌ Admin user not found. Please run the setup script first.",
      );
      return;
    }

    console.log("👤 Found admin user:", adminUser.email);

    // Create sample cases
    console.log("📁 Creating sample cases...");
    const sampleCases = await db
      .insert(schema.cases)
      .values([
        {
          caseNumber: "CASE-2025-001",
          title: "State v. Johnson - Armed Robbery",
          description:
            "Armed robbery of convenience store with multiple witnesses and security footage.",
          incidentDate: new Date("2025-01-15"),
          location: "123 Main Street, Downtown",
          priority: "high",
          status: "open",
          category: "violent_crime",
          dangerScore: 8,
          estimatedValue: 2500.0,
          jurisdiction: "State Court",
          leadProsecutor: adminUser.id,
          aiSummary:
            "High-priority armed robbery case with strong evidence including video surveillance and witness testimony.",
          aiTags: [
            "armed_robbery",
            "violent_crime",
            "video_evidence",
            "witnesses",
          ],
          metadata: {
            priority_reason: "violent crime with weapon",
            evidence_strength: "strong",
          },
          createdBy: adminUser.id,
        },
        {
          caseNumber: "CASE-2025-002",
          title: "People v. Smith - Fraud Investigation",
          description:
            "Identity theft and credit card fraud investigation involving multiple victims.",
          incidentDate: new Date("2025-01-20"),
          location: "Various locations",
          priority: "medium",
          status: "investigation",
          category: "financial_crime",
          dangerScore: 3,
          estimatedValue: 15000.0,
          jurisdiction: "Federal Court",
          leadProsecutor: adminUser.id,
          aiSummary:
            "Complex financial fraud case requiring extensive digital forensics and victim coordination.",
          aiTags: [
            "fraud",
            "identity_theft",
            "financial_crime",
            "digital_evidence",
          ],
          metadata: { victim_count: 12, investigation_complexity: "high" },
          createdBy: adminUser.id,
        },
        {
          caseNumber: "CASE-2025-003",
          title: "State v. Williams - Drug Distribution",
          description:
            "Large-scale drug distribution operation with multiple defendants.",
          incidentDate: new Date("2025-02-01"),
          location: "456 Oak Avenue, Eastside",
          priority: "high",
          status: "open",
          category: "drug_crime",
          dangerScore: 7,
          estimatedValue: 50000.0,
          jurisdiction: "State Court",
          leadProsecutor: adminUser.id,
          aiSummary:
            "Multi-defendant drug distribution case with surveillance evidence and undercover operations.",
          aiTags: [
            "drug_distribution",
            "organized_crime",
            "surveillance",
            "undercover",
          ],
          metadata: { defendant_count: 5, operation_duration: "6_months" },
          createdBy: adminUser.id,
        },
      ])
      .returning();

    console.log(`✅ Created ${sampleCases.length} sample cases`);

    // Create sample evidence for each case
    console.log("🔍 Creating sample evidence...");
    const sampleEvidence = [];

    for (const caseItem of sampleCases) {
      if (caseItem.title.includes("Johnson")) {
        // Evidence for armed robbery case
        const evidence = await db
          .insert(schema.evidence)
          .values([
            {
              caseId: caseItem.id,
              title: "Security Camera Footage",
              description:
                "HD security footage showing suspect entering store with weapon",
              evidenceType: "video",
              fileType: "mp4",
              fileName: "store_security_20250115.mp4",
              fileSize: 245760000,
              mimeType: "video/mp4",
              tags: [
                "security_footage",
                "suspect_identification",
                "weapon_visible",
              ],
              collectedAt: new Date("2025-01-15T10:30:00"),
              collectedBy: "Detective Martinez",
              location: "QuickMart Security System",
              aiSummary:
                "Clear footage shows suspect brandishing firearm and demanding cash from register.",
              aiTags: ["weapon", "robbery", "clear_identification"],
              isAdmissible: true,
              confidentialityLevel: "restricted",
              uploadedBy: adminUser.id,
            },
            {
              caseId: caseItem.id,
              title: "Witness Statement - Store Clerk",
              description:
                "Detailed witness statement from store clerk who was present during robbery",
              evidenceType: "document",
              fileType: "pdf",
              fileName: "witness_statement_clerk.pdf",
              fileSize: 1024000,
              mimeType: "application/pdf",
              tags: ["witness_statement", "victim_testimony", "eyewitness"],
              collectedAt: new Date("2025-01-15T14:00:00"),
              collectedBy: "Officer Johnson",
              location: "Police Station",
              aiSummary:
                "Clerk provides detailed description of suspect and sequence of events.",
              aiTags: ["witness", "testimony", "suspect_description"],
              isAdmissible: true,
              confidentialityLevel: "standard",
              uploadedBy: adminUser.id,
            },
            {
              caseId: caseItem.id,
              title: "Recovered Weapon",
              description:
                "Smith & Wesson revolver recovered from suspect during arrest",
              evidenceType: "physical",
              subType: "weapon",
              tags: ["weapon", "firearm", "physical_evidence"],
              collectedAt: new Date("2025-01-16T08:00:00"),
              collectedBy: "Detective Rodriguez",
              location: "Suspect residence",
              chainOfCustody: [
                {
                  officer: "Detective Rodriguez",
                  timestamp: "2025-01-16T08:00:00",
                  action: "collected",
                },
                {
                  officer: "Evidence Tech",
                  timestamp: "2025-01-16T10:00:00",
                  action: "logged",
                },
              ],
              labAnalysis: {
                fingerprints: "match_suspect",
                ballistics: "pending",
              },
              aiSummary:
                "Firearm matches description from witness statements and security footage.",
              aiTags: ["weapon", "fingerprint_match", "ballistics"],
              isAdmissible: true,
              confidentialityLevel: "restricted",
              uploadedBy: adminUser.id,
            },
          ])
          .returning();
        sampleEvidence.push(...evidence);
      }

      if (caseItem.title.includes("Smith")) {
        // Evidence for fraud case
        const evidence = await db
          .insert(schema.evidence)
          .values([
            {
              caseId: caseItem.id,
              title: "Bank Transaction Records",
              description:
                "Fraudulent credit card transactions spanning 3 months",
              evidenceType: "digital",
              fileType: "xlsx",
              fileName: "fraudulent_transactions.xlsx",
              fileSize: 2048000,
              mimeType:
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
              tags: ["financial_records", "fraud_evidence", "banking"],
              collectedAt: new Date("2025-01-21T09:00:00"),
              collectedBy: "Financial Crimes Unit",
              location: "Bank of America",
              aiSummary:
                "Pattern of fraudulent activity across multiple victim accounts.",
              aiTags: [
                "fraud_pattern",
                "financial_evidence",
                "multiple_victims",
              ],
              isAdmissible: true,
              confidentialityLevel: "confidential",
              uploadedBy: adminUser.id,
            },
            {
              caseId: caseItem.id,
              title: "Suspect Computer Hard Drive",
              description: "Encrypted hard drive seized from suspect residence",
              evidenceType: "digital",
              subType: "computer",
              tags: ["digital_evidence", "computer_forensics", "encrypted"],
              collectedAt: new Date("2025-01-22T16:00:00"),
              collectedBy: "Cyber Crimes Unit",
              location: "Suspect residence",
              labAnalysis: {
                encryption: "AES-256",
                status: "decryption_in_progress",
                preliminary_findings: "victim_data_found",
              },
              aiSummary:
                "Computer contains evidence of identity theft and fraud operations.",
              aiTags: ["digital_forensics", "identity_theft", "victim_data"],
              isAdmissible: true,
              confidentialityLevel: "restricted",
              uploadedBy: adminUser.id,
            },
          ])
          .returning();
        sampleEvidence.push(...evidence);
      }
    }

    console.log(`✅ Created ${sampleEvidence.length} pieces of evidence`);

    // Create sample reports
    console.log("📄 Creating sample reports...");
    const sampleReports = await db
      .insert(schema.reports)
      .values([
        {
          title: "Preliminary Investigation Report - Johnson Case",
          content: {
            summary:
              "Initial investigation into armed robbery at QuickMart on Main Street",
            sections: [
              {
                title: "Incident Overview",
                content:
                  "On January 15, 2025, at approximately 10:15 AM, an armed robbery occurred at QuickMart convenience store located at 123 Main Street. The suspect entered the store, displayed a firearm, and demanded cash from the register.",
              },
              {
                title: "Evidence Collected",
                content:
                  "Security footage, witness statements, and the weapon used in the crime have been secured and are being processed by the appropriate units.",
              },
              {
                title: "Next Steps",
                content:
                  "Ballistics analysis pending, additional witness interviews scheduled, and suspect background investigation ongoing.",
              },
            ],
          },
          summary:
            "Preliminary investigation findings for armed robbery case with strong evidence and witness cooperation.",
          caseId: sampleCases[0].id,
          reportType: "investigation",
          type: "investigation",
          status: "draft",
          confidentialityLevel: "restricted",
          jurisdiction: "State Court",
          tags: ["preliminary", "armed_robbery", "investigation"],
          aiSummary:
            "Strong case with multiple evidence types and cooperative witnesses.",
          wordCount: 250,
          estimatedReadTime: 2,
          createdBy: adminUser.id,
        },
        {
          title: "Financial Crimes Analysis - Smith Investigation",
          content: {
            summary:
              "Comprehensive analysis of financial fraud scheme affecting multiple victims",
            sections: [
              {
                title: "Fraud Scheme Overview",
                content:
                  "Investigation reveals sophisticated identity theft operation targeting elderly victims through phishing emails and fake websites.",
              },
              {
                title: "Victim Impact",
                content:
                  "Twelve confirmed victims with total losses exceeding $15,000. Additional victims may come forward as investigation progresses.",
              },
              {
                title: "Digital Evidence",
                content:
                  "Suspect computer contains victim personal information, fraudulent documents, and communication records with accomplices.",
              },
            ],
          },
          summary:
            "Detailed analysis of multi-victim fraud scheme with digital evidence and victim statements.",
          caseId: sampleCases[1].id,
          reportType: "analysis",
          type: "analysis",
          status: "review",
          confidentialityLevel: "confidential",
          jurisdiction: "Federal Court",
          tags: ["financial_crimes", "fraud_analysis", "digital_evidence"],
          aiSummary:
            "Complex fraud case requiring coordination with federal agencies and victim services.",
          wordCount: 420,
          estimatedReadTime: 3,
          createdBy: adminUser.id,
        },
      ])
      .returning();

    console.log(`✅ Created ${sampleReports.length} sample reports`);

    // Create some saved citations
    console.log("📚 Creating sample citations...");
    const sampleCitations = await db
      .insert(schema.savedCitations)
      .values([
        {
          userId: adminUser.id,
          title: "Armed Robbery Statute",
          content:
            "Any person who commits robbery while armed with a dangerous weapon shall be punished by imprisonment for not less than five years.",
          source: "State Penal Code § 211(a)",
          tags: ["armed_robbery", "sentencing", "weapons"],
          category: "statutes",
          isFavorite: true,
          contextData: {
            section: "211(a)",
            chapter: "Robbery",
            severity: "felony",
          },
        },
        {
          userId: adminUser.id,
          title: "Identity Theft Definition",
          content:
            "Identity theft occurs when someone uses another person's personal identifying information without permission to commit fraud or other crimes.",
          source: "Federal Identity Theft Act § 1028",
          tags: ["identity_theft", "fraud", "federal_law"],
          category: "statutes",
          isFavorite: false,
          contextData: { jurisdiction: "federal", penalties: "up_to_15_years" },
        },
        {
          userId: adminUser.id,
          title: "Chain of Custody Requirements",
          content:
            "Evidence must be properly documented and maintained through an unbroken chain of custody to be admissible in court proceedings.",
          source: "Evidence Code § 1400-1402",
          tags: ["evidence", "chain_of_custody", "admissibility"],
          category: "evidence",
          isFavorite: true,
          contextData: { importance: "critical", area: "evidence_law" },
        },
      ])
      .returning();

    console.log(`✅ Created ${sampleCitations.length} sample citations`);

    console.log("\n🎉 Sample data seeding completed successfully!");
    console.log(`
📊 Summary:
   • ${sampleCases.length} Cases created
   • ${sampleEvidence.length} Evidence items added
   • ${sampleReports.length} Reports generated
   • ${sampleCitations.length} Citations saved
   
🔗 Cases created:
   • ${sampleCases[0].caseNumber}: ${sampleCases[0].title}
   • ${sampleCases[1].caseNumber}: ${sampleCases[1].title}
   • ${sampleCases[2].caseNumber}: ${sampleCases[2].title}
`);
  } catch (error) {
    console.error("❌ Error seeding sample data:", error);
  } finally {
    await pool.end();
  }
}

seedSampleData();
