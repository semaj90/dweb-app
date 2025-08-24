
import { db } from './drizzle';
import { users, userProfiles, cases, evidence, criminals } from './schema-postgres';
import bcrypt from 'bcryptjs';
import { generateId } from 'lucia';

// Generate unique IDs for consistent seeding
function generateIdFromEntropySize(size: number): string {
  return generateId(size);
}

async function seed() {
  console.log('🌱 Starting database seed...');

  try {
    // Create test users
    console.log('👤 Creating users...');
    const passwordHash = await bcrypt.hash('demo123456', 10);

    const seedUsers = [
      {
        email: 'demo@legalai.gov',
        hashedPassword: passwordHash,
        firstName: 'Demo',
        lastName: 'Prosecutor', 
        name: 'Demo Prosecutor',
        role: 'prosecutor',
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date()
      },
      {
        email: 'detective@legalai.gov',
        hashedPassword: passwordHash,
        firstName: 'Jane',
        lastName: 'Detective',
        name: 'Jane Detective', 
        role: 'investigator',
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date()
      },
      {
        email: 'admin@legalai.gov',
        hashedPassword: passwordHash,
        firstName: 'System',
        lastName: 'Administrator',
        name: 'System Administrator',
        role: 'admin',
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date()
      }
    ];

    const insertedUsers = await db.insert(users).values(seedUsers).returning();
    console.log(`✅ Created ${insertedUsers.length} users`);

    // Create test cases
    console.log('📁 Creating cases...');
    const seedCases = [
      {
        caseNumber: 'DEMO-2024-001',
        title: 'Corporate Fraud Investigation - TechCorp Inc.',
        name: 'People vs. TechCorp Inc.',
        description: 'Investigation into alleged securities fraud and financial misrepresentation by TechCorp Inc. executives. Case involves complex financial documents and requires AI-assisted analysis.',
        incidentDate: new Date('2024-01-15'),
        location: 'San Francisco, CA',
        priority: 'high',
        status: 'open',
        category: 'white_collar_crime',
        dangerScore: 3,
        estimatedValue: '2500000.00',
        jurisdiction: 'Superior Court of California, San Francisco County',
        leadProsecutor: insertedUsers[0].id,
        assignedTeam: [insertedUsers[0].id],
        tags: ['securities_fraud', 'financial_crime', 'corporate', 'high_profile'],
        aiSummary: 'This case involves complex financial fraud allegations against TechCorp Inc. AI analysis suggests strong documentary evidence and potential for successful prosecution.',
        aiTags: ['complex_financial_documents', 'requires_expert_testimony', 'high_media_attention'],
        metadata: {
          estimatedTrialDate: '2024-08-15',
          publicInterest: 'high',
          mediaRestrictions: true,
          specialRequirements: ['financial_expert', 'forensic_accountant']
        },
        createdBy: insertedUsers[0].id,
        createdAt: new Date(),
        updatedAt: new Date()
      },
      {
        caseNumber: 'DEMO-2024-002',
        title: 'Cybercrime Investigation - Data Breach',
        name: 'State vs. Unknown Hackers',
        description: 'Large-scale data breach affecting over 100,000 residents. Investigation into advanced persistent threat group targeting government systems.',
        incidentDate: new Date('2024-02-01'),
        location: 'Sacramento, CA',
        priority: 'high',
        status: 'open',
        category: 'cybercrime',
        dangerScore: 4,
        jurisdiction: 'California Attorney General\'s Office',
        leadProsecutor: insertedUsers[1].id,
        assignedTeam: [insertedUsers[1].id, insertedUsers[0].id],
        tags: ['data_breach', 'cybercrime', 'apt_group', 'government_target'],
        aiSummary: 'Sophisticated cyber attack with international connections. Evidence suggests state-sponsored activity.',
        aiTags: ['international_threat', 'requires_federal_cooperation', 'technical_complexity'],
        createdBy: insertedUsers[1].id,
        createdAt: new Date(),
        updatedAt: new Date()
      }
    ];

    const insertedCases = await db.insert(cases).values(seedCases).returning();
    console.log(`✅ Created ${insertedCases.length} cases`);

    // Create sample criminal
    console.log('👤 Creating criminals...');
    const sampleCriminal = await db
      .insert(criminals)
      .values({
        firstName: 'John',
        lastName: 'Executive',
        middleName: 'Michael',
        aliases: ['J.M. Executive', 'John M. Smith'],
        dateOfBirth: new Date('1975-03-20'),
        placeOfBirth: 'New York, NY',
        address: '456 Executive Boulevard, Palo Alto, CA 94301',
        phone: '+1 (555) 987-6543',
        email: 'j.executive@techcorp.com',
        height: 180,
        weight: 75,
        eyeColor: 'brown',
        hairColor: 'black',
        distinguishingMarks: 'Small scar on left hand, corporate tattoo on right wrist',
        threatLevel: 'medium',
        status: 'active',
        notes: 'Former CEO of TechCorp Inc. Sophisticated white-collar criminal with extensive financial knowledge. Flight risk due to international connections.',
        aiSummary: 'High-profile executive with complex financial background. AI analysis indicates pattern of sophisticated fraud schemes.',
        aiTags: ['flight_risk', 'sophisticated_criminal', 'financial_expertise', 'international_connections'],
        createdBy: insertedUsers[0].id,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      .returning();

    console.log('✅ Created sample criminal');

    // Create test evidence
    console.log('🔍 Creating evidence...');
    const seedEvidence = [
      {
        caseId: insertedCases[0].id,
        criminalId: sampleCriminal[0].id,
        title: 'Financial Records - Q4 2023',
        description: 'Comprehensive financial statements and internal accounting documents from TechCorp Inc. for Q4 2023, showing discrepancies in reported revenue.',
        evidenceType: 'document',
        fileType: 'pdf',
        subType: 'financial_records',
        fileName: 'techcorp_q4_2023_financials.pdf',
        fileSize: 2457600,
        mimeType: 'application/pdf',
        hash: 'sha256:a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
        tags: ['financial_statements', 'revenue_discrepancy', 'q4_2023', 'key_evidence'],
        chainOfCustody: [
          {
            timestamp: new Date('2024-02-01T10:30:00Z'),
            action: 'collected',
            officer: 'Detective Sarah Johnson',
            location: 'TechCorp Inc. Headquarters',
            notes: 'Collected during authorized search warrant execution'
          }
        ],
        collectedAt: new Date('2024-02-01T10:30:00Z'),
        collectedBy: 'Detective Sarah Johnson',
        location: 'TechCorp Inc. Headquarters, 789 Tech Drive, San Francisco, CA',
        aiAnalysis: {
          confidence: 0.92,
          keyFindings: [
            'Revenue inflation detected in Q4 figures',
            'Inconsistent accounting methods applied',
            'Potential tax evasion indicators'
          ]
        },
        aiTags: ['revenue_fraud', 'accounting_irregularities', 'expert_testimony_required'],
        aiSummary: 'Critical financial evidence showing systematic revenue inflation. AI analysis indicates high probability of intentional fraud.',
        summary: 'Key documentary evidence in the TechCorp fraud case. Contains financial discrepancies that support prosecution theory.',
        isAdmissible: true,
        confidentialityLevel: 'restricted',
        uploadedBy: insertedUsers[0].id,
        uploadedAt: new Date('2024-02-01T14:15:00Z'),
        updatedAt: new Date('2024-02-01T14:15:00Z')
      },
      {
        caseId: insertedCases[0].id,
        title: 'Email Communications - Executive Team',
        description: 'Internal email thread between executives discussing "creative accounting" practices.',
        evidenceType: 'digital',
        fileType: 'email',
        subType: 'communication_records',
        fileName: 'executive_emails_jan2024.mbox',
        fileSize: 1024000,
        mimeType: 'application/mbox',
        hash: 'sha256:b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567',
        tags: ['email_evidence', 'internal_communications', 'executive_intent'],
        collectedAt: new Date('2024-02-05T09:20:00Z'),
        collectedBy: 'Digital Forensics Team',
        aiSummary: 'Email evidence shows awareness of fraudulent practices among executive team.',
        isAdmissible: true,
        confidentialityLevel: 'restricted',
        uploadedBy: insertedUsers[0].id,
        uploadedAt: new Date('2024-02-05T09:20:00Z'),
        updatedAt: new Date('2024-02-05T09:20:00Z')
      },
      {
        caseId: insertedCases[1].id,
        title: 'Server Access Logs',
        description: 'System logs showing unauthorized database access and data exfiltration attempts.',
        evidenceType: 'digital',
        fileType: 'log',
        subType: 'system_logs',
        fileName: 'server_access_logs_feb2024.txt',
        fileSize: 567000,
        mimeType: 'text/plain',
        hash: 'sha256:c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678',
        tags: ['system_logs', 'unauthorized_access', 'data_breach'],
        collectedAt: new Date('2024-02-10T11:45:00Z'),
        collectedBy: 'Cybercrime Unit',
        aiSummary: 'Server logs reveal systematic unauthorized access and data exfiltration over 2-week period.',
        isAdmissible: true,
        confidentialityLevel: 'classified',
        uploadedBy: insertedUsers[1].id,
        uploadedAt: new Date('2024-02-10T11:45:00Z'),
        updatedAt: new Date('2024-02-10T11:45:00Z')
      }
    ];

    const insertedEvidence = await db.insert(evidence).values(seedEvidence).returning();
    console.log(`✅ Created ${insertedEvidence.length} evidence items`);

    // Create user profiles for demo users
    console.log('📝 Creating user profiles...');
    await db
      .insert(userProfiles)
      .values({
        userId: insertedUsers[0].id,
        bio: 'Demo prosecutor account for testing the Legal AI Platform',
        phone: '+1 (555) 123-4567',
        address: '123 Justice Drive, Legal City, LC 12345',
        preferences: {
          theme: 'yorha',
          notifications: true,
          aiAssistance: true
        },
        permissions: [
          'case_create',
          'case_edit',
          'case_view',
          'evidence_upload',
          'evidence_analyze',
          'ai_query',
          'report_generate'
        ],
        specializations: [
          'white_collar_crime',
          'corporate_fraud',
          'financial_investigations'
        ],
        experienceLevel: 'senior',
        workPatterns: {
          mostActiveHours: [8, 9, 10, 11, 13, 14, 15, 16],
          documentsPerWeek: 25,
          casesHandled: 12
        },
        metadata: {
          jurisdiction: 'State of California',
          preferredModel: 'gemma3-legal'
        },
        createdAt: new Date(),
        updatedAt: new Date()
      });

    console.log('✅ Created demo user profile');

    console.log('\n🎉 Database seed completed successfully!');
    console.log(`
📊 Summary:
   👤 Users: ${insertedUsers.length}
   📁 Cases: ${insertedCases.length} 
   🔍 Evidence: ${insertedEvidence.length}
   🚨 Criminals: 1

🔐 Demo Login Credentials:
   demo@legalai.gov / demo123456
   detective@legalai.gov / demo123456
   admin@legalai.gov / demo123456

📁 Sample Case: DEMO-2024-001 - Corporate Fraud Investigation
   • TechCorp Inc. securities fraud case
   • Complex financial documents requiring AI analysis
   • 3 evidence items with detailed chain of custody
   • Linked to John Executive criminal profile
`);

  } catch (error) {
    console.error('❌ Seed failed:', error);
    throw error;
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  seed();
}

export { seed };