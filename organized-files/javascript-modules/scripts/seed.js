import { db } from "../src/lib/server/db/index.js";
import {
  users,
  knowledgeBase,
  cases,
  chatSessions,
} from "../src/lib/server/db/schema.js";
import { embeddingService } from "../src/lib/server/embedding.js";

/**
 * Database seeding script for legal AI system
 * Populates the database with sample users, cases, and legal knowledge
 */

async function seedDatabase() {
  console.log("🌱 Starting database seeding...");

  try {
    // Create sample users
    console.log("👥 Creating sample users...");
    const [sampleUser] = await db
      .insert(users)
      .values({
        email: "prosecutor@legal.gov",
        name: "Sarah Johnson",
        role: "prosecutor",
        preferences: {
          theme: "dark",
          notifications: true,
          autoSave: true,
        },
      })
      .returning();

    console.log(`✅ Created user: ${sampleUser.name}`);

    // Create sample case
    console.log("📁 Creating sample case...");
    const [sampleCase] = await db
      .insert(cases)
      .values({
        caseNumber: "CASE-2024-001",
        title: "State v. Anderson - Embezzlement Investigation",
        description:
          "Corporate embezzlement case involving financial fraud and document forgery.",
        status: "active",
        priority: "high",
        assignedTo: sampleUser.id,
        createdBy: sampleUser.id,
        metadata: {
          estimatedValue: 2500000,
          defendants: ["Robert Anderson"],
          witnesses: 8,
          documents: 150,
        },
      })
      .returning();

    console.log(`✅ Created case: ${sampleCase.title}`);

    // Create sample chat session
    console.log("💬 Creating sample chat session...");
    const [sampleSession] = await db
      .insert(chatSessions)
      .values({
        userId: sampleUser.id,
        title: "Evidence Analysis Discussion",
        caseId: sampleCase.id,
        metadata: {
          tags: ["evidence", "analysis", "financial"],
          priority: "high",
        },
      })
      .returning();

    console.log(`✅ Created chat session: ${sampleSession.title}`);

    // Sample legal knowledge base entries
    console.log("📚 Populating legal knowledge base...");

    const knowledgeEntries = [
      {
        title: "Elements of Embezzlement",
        content: `Embezzlement requires proof of the following elements:

1. **Fiduciary Relationship**: The defendant must have been in a position of trust with access to the victim's property or funds.

2. **Lawful Possession**: The defendant must have had lawful possession or control of the property initially.

3. **Conversion**: The defendant must have converted the property to their own use or benefit.

4. **Intent to Deprive**: The defendant must have intended to permanently deprive the owner of the property.

5. **Value**: Most jurisdictions require proof of the monetary value of the embezzled property to determine the degree of the offense.

**Key Case Law**: In *People v. Davis* (2019), the court held that temporary use with intent to return does not constitute embezzlement if there was no intent to permanently deprive.

**Prosecutorial Considerations**:
- Document the chain of custody for financial records
- Establish clear timeline of defendant's access and control
- Gather evidence of personal benefit or use
- Consider forensic accounting for complex cases`,
        type: "statute",
        jurisdiction: "General",
        citation: "18 U.S.C. § 641; State Penal Codes Vary",
        keyTerms: [
          "embezzlement",
          "fiduciary",
          "conversion",
          "intent",
          "property",
        ],
      },
      {
        title: "Document Authentication Standards",
        content: `Federal Rules of Evidence for Document Authentication:

**Rule 901 - Authenticating or Identifying Evidence**

To satisfy the requirement of authenticating or identifying an item of evidence, the proponent must produce evidence sufficient to support a finding that the item is what the proponent claims it is.

**Common Methods of Authentication**:

1. **Testimony of Witness with Knowledge** (901(b)(1))
   - Direct observation of creation or execution
   - Personal knowledge of the document's history

2. **Nonexpert Opinion About Handwriting** (901(b)(2))
   - Familiarity with handwriting not acquired for litigation

3. **Comparison by Expert or Trier of Fact** (901(b)(3))
   - Expert comparison with authenticated specimens
   - Jury comparison with authenticated specimens

4. **Distinctive Characteristics** (901(b)(4))
   - Appearance, contents, substance, internal patterns
   - Circumstantial evidence of authenticity

5. **Opinion About a Voice** (901(b)(5))
   - Voice identification based on hearing the voice

**Digital Evidence Considerations**:
- Hash values for data integrity verification
- Chain of custody documentation
- Metadata preservation and analysis
- Expert testimony for complex digital formats

**Best Practices**:
- Maintain detailed custody logs
- Use certified forensic imaging tools
- Document collection procedures thoroughly
- Consider self-authentication options under Rule 902`,
        type: "procedure",
        jurisdiction: "Federal",
        citation: "Fed. R. Evid. 901",
        keyTerms: [
          "authentication",
          "evidence",
          "documents",
          "chain of custody",
          "digital evidence",
        ],
      },
      {
        title: "Witness Preparation Guidelines",
        content: `Ethical Guidelines for Witness Preparation in Criminal Cases:

**Permissible Preparation Activities**:

1. **Factual Review**
   - Review known facts and evidence with witness
   - Discuss witness's personal knowledge and observations
   - Clarify timeline and sequence of events

2. **Procedure Explanation**
   - Explain courtroom procedures and expectations
   - Describe direct and cross-examination process
   - Review oath-taking responsibilities

3. **Communication Training**
   - Practice clear, audible speech
   - Emphasize importance of listening to questions
   - Train to ask for clarification when needed

4. **Truthfulness Emphasis**
   - Stress absolute requirement for truthful testimony
   - Explain consequences of perjury
   - Encourage admission of uncertainty or lack of memory

**Prohibited Activities**:
- Suggesting false or misleading testimony
- Encouraging witness to change testimony
- Discussing testimony with other witnesses inappropriately
- Coaching on substantive answers beyond known facts

**Special Considerations for Vulnerable Witnesses**:
- Child witnesses may require specialized preparation
- Trauma-informed approaches for crime victims
- Language interpreters and cultural considerations
- Accommodations for disabilities

**Documentation Requirements**:
- Record preparation sessions appropriately
- Note any concerns about witness reliability
- Document any limitations in witness knowledge
- Maintain privilege where applicable

**Model Rule 3.4 Compliance**:
Prosecutors must not unlawfully obstruct access to evidence or unlawfully alter, destroy, or conceal evidence, nor counsel or assist a witness to testify falsely.`,
        type: "procedure",
        jurisdiction: "General",
        citation: "Model Rules of Professional Conduct 3.4",
        keyTerms: [
          "witness preparation",
          "ethics",
          "testimony",
          "cross-examination",
          "truthfulness",
        ],
      },
      {
        title: "Brady Material Disclosure Requirements",
        content: `Brady v. Maryland (1963) - Prosecutorial Disclosure Obligations:

**Constitutional Foundation**:
The Due Process Clause requires prosecutors to disclose material evidence favorable to the defense, whether it relates to guilt or punishment.

**Three-Part Brady Test**:
1. **Favorable Evidence**: Evidence must be favorable to the accused
2. **Material Evidence**: Evidence must be material to guilt or punishment
3. **Prejudice**: Suppression must have prejudiced the defendant

**Categories of Brady Material**:

1. **Exculpatory Evidence**
   - Evidence that tends to prove defendant's innocence
   - Alternative suspect information
   - Evidence undermining prosecution theories

2. **Impeachment Evidence (Giglio)**
   - Evidence affecting witness credibility
   - Witness plea agreements or immunity deals
   - Prior inconsistent statements
   - Criminal history of prosecution witnesses

3. **Mitigation Evidence**
   - Evidence relevant to sentencing
   - Mental health evaluations
   - Background and character evidence

**Timing Requirements**:
- Generally before trial, but specifics vary by jurisdiction
- Some evidence may be disclosed during trial if newly discovered
- Capital cases often have stricter timing requirements

**Practical Considerations**:
- Maintain open file policies where possible
- Regular communication with defense counsel
- Thorough review of all case materials
- Training for law enforcement on Brady obligations

**Sanctions for Violations**:
- Dismissal of charges (rare)
- New trial (most common remedy)
- Exclusion of evidence
- Judicial sanctions against prosecutor

**Best Practices**:
- Err on the side of disclosure
- Document disclosure decisions
- Establish clear procedures with law enforcement
- Regular Brady training for prosecution staff`,
        type: "case_law",
        jurisdiction: "Federal",
        citation: "Brady v. Maryland, 373 U.S. 83 (1963)",
        keyTerms: [
          "Brady",
          "disclosure",
          "exculpatory",
          "due process",
          "material evidence",
        ],
      },
      {
        title: "Search Warrant Affidavit Requirements",
        content: `Fourth Amendment - Search Warrant Standards:

**Constitutional Standard**:
"The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated, and no Warrants shall issue, but upon probable cause..."

**Probable Cause Elements**:
1. **Reliability of Information**
   - Veracity of informant or source
   - Basis of knowledge of informant
   - Independent police corroboration

2. **Nexus Requirements**
   - Evidence likely to be found at location
   - Connection between crime and place to be searched
   - Timeliness of information (staleness doctrine)

**Affidavit Components**:

1. **Identity and Qualifications**
   - Affiant's name, rank, and experience
   - Specialized training relevant to investigation
   - Previous experience with similar cases

2. **Statement of Facts**
   - Detailed description of criminal activity
   - Specific facts supporting probable cause
   - Source of information and reliability indicators

3. **Description of Place to be Searched**
   - Precise physical address or description
   - Specific areas or structures included
   - Limitations on scope of search

4. **Description of Items to be Seized**
   - Specific evidence relating to crime
   - Contraband or instrumentalities of crime
   - Avoid overly broad or general descriptions

**Special Warrant Types**:

- **Anticipatory Warrants**: Based on triggering condition
- **Digital Evidence Warrants**: Special considerations for electronic data
- **No-Knock Warrants**: Enhanced safety or evidence preservation requirements

**Common Deficiencies to Avoid**:
- Stale information without recent corroboration
- Insufficient nexus between crime and search location
- Overly broad descriptions of items to be seized
- Failure to establish informant reliability
- Boilerplate language without case-specific facts

**Franks Hearing Considerations**:
Defendants may challenge warrant affidavits for material omissions or false statements made knowingly or with reckless disregard for truth.`,
        type: "procedure",
        jurisdiction: "Federal",
        citation: "U.S. Const. amend. IV; Fed. R. Crim. P. 41",
        keyTerms: [
          "search warrant",
          "probable cause",
          "Fourth Amendment",
          "affidavit",
          "nexus",
        ],
      },
    ];

    // Process and store knowledge base entries
    for (const entry of knowledgeEntries) {
      console.log(`📄 Processing: ${entry.title}`);

      // Generate embedding for the content
      const embedding = await embeddingService.generateEmbedding(
        `${entry.title}\n\n${entry.content}`
      );

      await db.insert(knowledgeBase).values({
        ...entry,
        embedding,
        createdBy: sampleUser.id,
      });

      console.log(`✅ Added: ${entry.title}`);
    }

    console.log("🎉 Database seeding completed successfully!");
    console.log("\n📊 Summary:");
    console.log(`- Users created: 1`);
    console.log(`- Cases created: 1`);
    console.log(`- Chat sessions created: 1`);
    console.log(`- Knowledge base entries: ${knowledgeEntries.length}`);
    console.log("\n🔗 Test the system at: http://localhost:5173/ai-test");
  } catch (error) {
    console.error("❌ Database seeding failed:", error);
    process.exit(1);
  }
}

// Check if this script is being run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  seedDatabase().then(() => process.exit(0));
}

export { seedDatabase };
