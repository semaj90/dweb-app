// Enhanced seed script with vector embeddings
import { db } from './sveltekit-frontend/src/lib/server/db';
import { users, cases, documents, evidence } from './sveltekit-frontend/src/lib/server/db/unified-schema';
import { VectorService } from './sveltekit-frontend/src/lib/server/services/vector.service';
import bcrypt from 'bcryptjs';

async function seed() {
  console.log('🌱 Starting database seed...');

  try {
    // Create demo users
    console.log('Creating users...');
    const hashedPassword = await bcrypt.hash('demo123', 12);
    
    const [adminUser] = await db.insert(users).values({
      email: 'admin@prosecutor.local',
      password: hashedPassword,
      role: 'admin',
    }).returning();

    const [regularUser] = await db.insert(users).values({
      email: 'user@prosecutor.local',
      password: hashedPassword,
      role: 'user',
    }).returning();

    console.log('✅ Users created');

    // Create demo cases
    console.log('Creating cases...');
    const demoCase1 = await db.insert(cases).values({
      caseNumber: 'CASE-2024-001',
      title: 'State v. Johnson - Contract Fraud',
      description: 'Defendant allegedly defrauded multiple investors through a Ponzi scheme involving real estate investments.',
      status: 'active',
      priority: 'high',
      assignedTo: adminUser.id,
      createdBy: adminUser.id,
      metadata: {
        jurisdiction: 'Federal',
        court: 'Northern District Court',
        judge: 'Hon. Sarah Williams',
        prosecutor: 'John Smith',
      },
    }).returning();

    const demoCase2 = await db.insert(cases).values({
      caseNumber: 'CASE-2024-002',
      title: 'People v. Rodriguez - Breach of Contract',
      description: 'Commercial dispute regarding failure to deliver goods as specified in purchase agreement.',
      status: 'active',
      priority: 'medium',
      assignedTo: regularUser.id,
      createdBy: adminUser.id,
      metadata: {
        jurisdiction: 'State',
        court: 'Superior Court',
        contractValue: '$250,000',
      },
    }).returning();

    console.log('✅ Cases created');

    // Create demo documents with embeddings
    console.log('Creating documents with embeddings...');
    
    const legalDocuments = [
      {
        title: 'Contract Analysis Report',
        content: `Legal Analysis: Contract Fraud Investigation
        
        This report examines the fraudulent activities in the Johnson case. The defendant systematically 
        misrepresented investment opportunities to victims, promising guaranteed returns of 15% monthly. 
        Evidence shows a clear pattern of deception, with funds from new investors being used to pay 
        earlier investors, characteristic of a Ponzi scheme.
        
        Key violations identified:
        1. Securities fraud under 15 U.S.C. § 78j(b)
        2. Wire fraud under 18 U.S.C. § 1343
        3. Money laundering under 18 U.S.C. § 1956
        
        The total amount defrauded exceeds $2.5 million across 47 victims.`,
        caseId: demoCase1[0].id,
        type: 'report' as const,
      },
      {
        title: 'Breach of Contract Legal Memorandum',
        content: `MEMORANDUM OF LAW
        
        RE: Rodriguez Contract Breach - Legal Remedies Available
        
        The plaintiff has clear grounds for breach of contract action. The defendant failed to deliver 
        10,000 units of specialized equipment by the contractually agreed date of January 15, 2024. 
        
        The contract explicitly states:
        "Time is of the essence. Failure to deliver by the specified date shall constitute material breach."
        
        Remedies available:
        1. Specific performance - Court may order delivery of goods
        2. Compensatory damages - Cover price differential and lost profits
        3. Consequential damages - Reasonably foreseeable losses from breach
        
        Recommendation: Pursue immediate injunctive relief to prevent disposal of goods.`,
        caseId: demoCase2[0].id,
        type: 'note' as const,
      },
      {
        title: 'Evidence Collection Protocol',
        content: `STANDARD OPERATING PROCEDURE: Digital Evidence Collection
        
        Purpose: Establish chain of custody for digital evidence in fraud investigations.
        
        Procedure:
        1. Document discovery - Record time, date, location, and discovering party
        2. Secure the evidence - Create forensic image before analysis
        3. Calculate hash values - Use SHA-256 for verification
        4. Maintain access log - Document all personnel who handle evidence
        5. Store in evidence locker - Use tamper-evident packaging
        
        Critical: Any break in chain of custody may result in evidence being ruled inadmissible.
        
        This protocol ensures compliance with Federal Rules of Evidence 901 regarding authentication.`,
        caseId: demoCase1[0].id,
        type: 'report' as const,
      },
    ];

    for (const doc of legalDocuments) {
      const [created] = await db.insert(documents).values({
        caseId: doc.caseId,
        title: doc.title,
        content: doc.content,
        documentType: doc.type,
        version: '1.0',
        createdBy: adminUser.id,
        metadata: {
          wordCount: doc.content.split(/\s+/).length,
          lastModified: new Date().toISOString(),
        },
      }).returning();

      // Store embeddings for semantic search
      try {
        await VectorService.storeDocument(
          created.id,
          doc.type,
          doc.content,
          { title: doc.title, caseId: doc.caseId }
        );
        console.log(`✅ Embeddings stored for: ${doc.title}`);
      } catch (error) {
        console.log(`⚠️  Could not store embeddings for ${doc.title} - Ollama may not be running`);
      }
    }

    // Create demo evidence
    console.log('Creating evidence entries...');
    
    await db.insert(evidence).values([
      {
        caseId: demoCase1[0].id,
        evidenceNumber: 'EV-2024-001-001',
        type: 'document',
        description: 'Bank statements showing money transfers',
        location: '/evidence/case-001/bank-statements.pdf',
        uploadedBy: adminUser.id,
        metadata: {
          fileSize: 2048576,
          mimeType: 'application/pdf',
          pages: 45,
        },
      },
      {
        caseId: demoCase1[0].id,
        evidenceNumber: 'EV-2024-001-002',
        type: 'email',
        description: 'Email correspondence between defendant and victims',
        location: '/evidence/case-001/emails.msg',
        uploadedBy: adminUser.id,
        metadata: {
          sender: 'johnson@fraudulent.com',
          recipients: ['victim1@email.com', 'victim2@email.com'],
          subject: 'Investment Opportunity',
        },
      },
      {
        caseId: demoCase2[0].id,
        evidenceNumber: 'EV-2024-002-001',
        type: 'document',
        description: 'Original purchase contract',
        location: '/evidence/case-002/contract.pdf',
        uploadedBy: regularUser.id,
        metadata: {
          fileSize: 512000,
          mimeType: 'application/pdf',
          signatureDate: '2023-11-01',
        },
      },
    ]);

    console.log('✅ Evidence created');

    console.log('\n🎉 Database seed completed successfully!');
    console.log('\nDemo accounts created:');
    console.log('  Admin: admin@prosecutor.local (password: demo123)');
    console.log('  User: user@prosecutor.local (password: demo123)');
    
  } catch (error) {
    console.error('❌ Seed failed:', error);
    throw error;
  }
}

// Run the seed
seed()
  .then(() => {
    console.log('\n✨ All done! You can now run: npm run dev');
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n💥 Fatal error:', error);
    process.exit(1);
  });
