import { db } from './pg.config';
import { users, cases, evidence, pois, comments } from './web-app/sveltekit-frontend/src/lib/schema';
import { hash } from 'bcryptjs';

/**
 * Seed the database with test data
 * Run with: npm run seed
 */
async function seed() {
  console.log('🌱 Seeding database...');

  try {
    // Create test users
    console.log('Creating test users...');
    
    const testUsers = await db.insert(users).values([
      {
        id: 'user_1',
        email: 'detective@warden-net.com',
        name: 'Detective Sarah Chen',
        role: 'detective',
        password_hash: await hash('password123', 10),
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'user_2', 
        email: 'admin@warden-net.com',
        name: 'Admin User',
        role: 'admin',
        password_hash: await hash('admin123', 10),
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'user_3',
        email: 'analyst@warden-net.com', 
        name: 'Crime Analyst Mike Rodriguez',
        role: 'analyst',
        password_hash: await hash('analyst123', 10),
        created_at: new Date(),
        updated_at: new Date()
      }
    ]).returning();

    // Create test cases
    console.log('Creating test cases...');
    
    const testCases = await db.insert(cases).values([
      {
        id: 'case_1',
        title: 'Downtown Burglary Series',
        description: 'Series of commercial burglaries in downtown district. Pattern indicates organized group targeting electronics stores.',
        status: 'active',
        priority: 'high',
        created_by: testUsers[0].id,
        assigned_to: testUsers[0].id,
        case_type: 'burglary',
        location: 'Downtown District',
        incident_date: new Date('2024-01-15'),
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'case_2',
        title: 'Riverside Park Vandalism',
        description: 'Multiple incidents of graffiti and property damage in Riverside Park area.',
        status: 'open',
        priority: 'medium', 
        created_by: testUsers[2].id,
        assigned_to: testUsers[0].id,
        case_type: 'vandalism',
        location: 'Riverside Park',
        incident_date: new Date('2024-01-20'),
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'case_3',
        title: 'Market Street Fraud Investigation',
        description: 'Financial fraud investigation involving multiple victims and complex money trail.',
        status: 'active',
        priority: 'high',
        created_by: testUsers[1].id,
        assigned_to: testUsers[2].id,
        case_type: 'fraud',
        location: 'Market Street Business District',
        incident_date: new Date('2024-01-10'),
        created_at: new Date(),
        updated_at: new Date()
      }
    ]).returning();

    // Create test evidence
    console.log('Creating test evidence...');
    
    const testEvidence = await db.insert(evidence).values([
      {
        id: 'evidence_1',
        case_id: testCases[0].id,
        title: 'Security Camera Footage - Store A',
        description: 'High-quality security footage showing suspects entering Store A',
        evidence_type: 'video',
        file_path: '/evidence/case_1/security_footage_store_a.mp4',
        file_size: 52428800, // 50MB
        mime_type: 'video/mp4',
        chain_of_custody: JSON.stringify([
          { officer: 'Detective Chen', date: '2024-01-15T10:30:00Z', action: 'Collected from scene' },
          { officer: 'Evidence Tech', date: '2024-01-15T14:00:00Z', action: 'Logged into evidence room' }
        ]),
        tags: ['security_footage', 'suspect_identification', 'store_a'],
        created_by: testUsers[0].id,
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'evidence_2',
        case_id: testCases[0].id,
        title: 'Fingerprints - Door Handle',
        description: 'Partial fingerprints lifted from rear door handle of Store B',
        evidence_type: 'forensic',
        file_path: '/evidence/case_1/fingerprints_door_handle.jpg',
        file_size: 2097152, // 2MB
        mime_type: 'image/jpeg',
        chain_of_custody: JSON.stringify([
          { officer: 'CSI Johnson', date: '2024-01-16T09:15:00Z', action: 'Collected from scene' },
          { officer: 'Forensics Lab', date: '2024-01-16T11:30:00Z', action: 'Processed for analysis' }
        ]),
        tags: ['fingerprints', 'forensic_evidence', 'store_b'],
        created_by: testUsers[0].id,
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'evidence_3',
        case_id: testCases[2].id,
        title: 'Bank Statements - Suspect Account',
        description: 'Bank statements showing suspicious transaction patterns',
        evidence_type: 'financial',
        file_path: '/evidence/case_3/bank_statements.pdf',
        file_size: 1048576, // 1MB
        mime_type: 'application/pdf',
        chain_of_custody: JSON.stringify([
          { officer: 'Analyst Rodriguez', date: '2024-01-11T13:45:00Z', action: 'Obtained via court order' },
          { officer: 'Financial Crimes Unit', date: '2024-01-11T16:00:00Z', action: 'Under analysis' }
        ]),
        tags: ['financial_records', 'bank_statements', 'fraud_evidence'],
        created_by: testUsers[2].id,
        created_at: new Date(),
        updated_at: new Date()
      }
    ]).returning();

    // Create test POIs (Points of Interest)
    console.log('Creating test POIs...');
    
    const testPOIs = await db.insert(pois).values([
      {
        id: 'poi_1',
        case_id: testCases[0].id,
        name: 'Electronics Store A',
        description: 'First target in burglary series. High-value electronics stolen.',
        poi_type: 'location',
        latitude: 40.7589,
        longitude: -73.9851,
        address: '123 Main Street, Downtown',
        significance: 'Primary crime scene',
        tags: ['crime_scene', 'burglary_target', 'electronics_store'],
        created_by: testUsers[0].id,
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'poi_2',
        case_id: testCases[0].id,
        name: 'Suspect Vehicle',
        description: 'Dark blue sedan spotted at multiple crime scenes',
        poi_type: 'vehicle',
        latitude: null,
        longitude: null,
        address: null,
        significance: 'Getaway vehicle - partial license plate XYZ-',
        tags: ['suspect_vehicle', 'getaway_car', 'blue_sedan'],
        created_by: testUsers[0].id,
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'poi_3',
        case_id: testCases[1].id,
        name: 'Riverside Park Main Entrance',
        description: 'Location of most severe vandalism incidents',
        poi_type: 'location',
        latitude: 40.7505,
        longitude: -73.9934,
        address: 'Riverside Park, Park Avenue Entrance',
        significance: 'Concentration of vandalism activity',
        tags: ['vandalism_hotspot', 'park_entrance', 'graffiti'],
        created_by: testUsers[2].id,
        created_at: new Date(),
        updated_at: new Date()
      }
    ]).returning();

    // Create test comments
    console.log('Creating test comments...');
    
    await db.insert(comments).values([
      {
        id: 'comment_1',
        case_id: testCases[0].id,
        user_id: testUsers[0].id,
        content: 'Security footage analysis reveals three suspects, all wearing dark clothing and masks. Working on facial recognition for unmasked moments.',
        comment_type: 'update',
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'comment_2',
        case_id: testCases[0].id,
        user_id: testUsers[1].id,
        content: 'Coordinating with neighboring precincts - similar MO reported in adjacent districts. Possible multi-jurisdiction case.',
        comment_type: 'coordination',
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'comment_3',
        case_id: testCases[2].id,
        user_id: testUsers[2].id,
        content: 'Financial analysis complete. Transaction patterns indicate money laundering through multiple shell companies. Preparing warrant requests.',
        comment_type: 'analysis',
        created_at: new Date(),
        updated_at: new Date()
      },
      {
        id: 'comment_4',
        case_id: testCases[1].id,
        user_id: testUsers[0].id,
        content: 'Witness interview scheduled for tomorrow. Local resident claims to have seen suspects multiple times.',
        comment_type: 'witness',
        created_at: new Date(),
        updated_at: new Date()
      }
    ]);

    console.log('✅ Database seeded successfully!');
    console.log(`Created:
    - ${testUsers.length} users
    - ${testCases.length} cases  
    - ${testEvidence.length} evidence items
    - ${testPOIs.length} POIs
    - 4 comments`);

    console.log('\n📋 Test Credentials:');
    console.log('Detective: detective@warden-net.com / password123');
    console.log('Admin: admin@warden-net.com / admin123');
    console.log('Analyst: analyst@warden-net.com / analyst123');

  } catch (error) {
    console.error('❌ Error seeding database:', error);
    throw error;
  }
}

// Run the seed function
if (require.main === module) {
  seed()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

export { seed };
