// Database Seed Script for YoRHa Interface
import { db } from '../src/lib/yorha/db';
import { units, achievements, missions, equipment } from '../src/lib/yorha/db/schema';
import bcrypt from 'bcryptjs';
import { nanoid } from 'nanoid';

async function seed() {
  console.log('🌱 Starting database seed...');
  
  try {
    // Seed achievements
    console.log('Creating achievements...');
    const achievementData = [
      // Mission achievements
      { name: 'First Mission', description: 'Complete your first mission', icon: '🎯', category: 'missions', requiredValue: 1, xpReward: 100 },
      { name: 'Mission Specialist', description: 'Complete 10 missions', icon: '⭐', category: 'missions', requiredValue: 10, xpReward: 500 },
      { name: 'Mission Commander', description: 'Complete 50 missions', icon: '🏆', category: 'missions', requiredValue: 50, xpReward: 2000 },
      { name: 'Mission Legend', description: 'Complete 100 missions', icon: '👑', category: 'missions', requiredValue: 100, xpReward: 5000 },
      
      // Level achievements
      { name: 'Level 5', description: 'Reach level 5', icon: '📈', category: 'level', requiredValue: 5, xpReward: 200 },
      { name: 'Level 10', description: 'Reach level 10', icon: '📊', category: 'level', requiredValue: 10, xpReward: 500 },
      { name: 'Level 25', description: 'Reach level 25', icon: '🎖️', category: 'level', requiredValue: 25, xpReward: 1500 },
      { name: 'Level 50', description: 'Reach level 50', icon: '🌟', category: 'level', requiredValue: 50, xpReward: 5000 },
      
      // Combat achievements
      { name: 'Combat Ready', description: 'Achieve 60% combat rating', icon: '⚔️', category: 'combat', requiredValue: 60, xpReward: 300 },
      { name: 'Combat Expert', description: 'Achieve 80% combat rating', icon: '🗡️', category: 'combat', requiredValue: 80, xpReward: 1000 },
      { name: 'Combat Master', description: 'Achieve 95% combat rating', icon: '💀', category: 'combat', requiredValue: 95, xpReward: 3000 },
      
      // Time achievements
      { name: 'Active Unit', description: 'Be active for 10 hours', icon: '⏰', category: 'time', requiredValue: 10, xpReward: 200 },
      { name: 'Dedicated Unit', description: 'Be active for 100 hours', icon: '📅', category: 'time', requiredValue: 100, xpReward: 1000 },
      { name: 'Veteran Unit', description: 'Be active for 500 hours', icon: '🎖️', category: 'time', requiredValue: 500, xpReward: 5000 },
      
      // Special achievements
      { name: 'Early Bird', description: 'Login before 6 AM', icon: '🌅', category: 'special', requiredValue: 1, xpReward: 100, hidden: true },
      { name: 'Night Owl', description: 'Login after midnight', icon: '🦉', category: 'special', requiredValue: 1, xpReward: 100, hidden: true },
      { name: 'Weekend Warrior', description: 'Complete missions on weekend', icon: '🎮', category: 'special', requiredValue: 1, xpReward: 200, hidden: true }
    ];
    
    for (const achievement of achievementData) {
      await db.insert(achievements).values(achievement).onConflictDoNothing();
    }
    console.log(`✅ Created ${achievementData.length} achievements`);
    
    // Seed missions
    console.log('Creating missions...');
    const missionData = [
      // Main missions
      {
        name: 'Operation: First Contact',
        description: 'Establish initial contact with the resistance camp',
        type: 'main',
        difficulty: 'easy',
        objectives: [
          { id: 1, description: 'Travel to the resistance camp', required: true },
          { id: 2, description: 'Speak with the leader', required: true }
        ],
        rewards: { xp: 500, credits: 100 },
        requiredLevel: 1,
        timeLimit: null
      },
      {
        name: 'Supply Run',
        description: 'Gather essential supplies for the resistance',
        type: 'main',
        difficulty: 'medium',
        objectives: [
          { id: 1, description: 'Collect 10 supply crates', required: true },
          { id: 2, description: 'Defeat hostile units', required: true },
          { id: 3, description: 'Return to base', required: true }
        ],
        rewards: { xp: 1000, credits: 250, items: ['Type-3 Sword'] },
        requiredLevel: 5,
        timeLimit: 3600
      },
      {
        name: 'Machine Nest Elimination',
        description: 'Clear out a machine nest threatening the area',
        type: 'main',
        difficulty: 'hard',
        objectives: [
          { id: 1, description: 'Locate the machine nest', required: true },
          { id: 2, description: 'Eliminate all hostile machines', required: true },
          { id: 3, description: 'Destroy the nest core', required: true }
        ],
        rewards: { xp: 2000, credits: 500, items: ['Type-40 Blade'] },
        requiredLevel: 10,
        timeLimit: 7200
      },
      
      // Side missions
      {
        name: 'Lost Signal',
        description: 'Investigate a mysterious signal in the desert',
        type: 'side',
        difficulty: 'medium',
        objectives: [
          { id: 1, description: 'Track the signal source', required: true },
          { id: 2, description: 'Investigate the anomaly', required: true }
        ],
        rewards: { xp: 750, credits: 150 },
        requiredLevel: 3,
        timeLimit: null
      },
      {
        name: 'Data Recovery',
        description: 'Recover lost data from corrupted terminals',
        type: 'side',
        difficulty: 'easy',
        objectives: [
          { id: 1, description: 'Access 5 terminals', required: true },
          { id: 2, description: 'Extract data fragments', required: true }
        ],
        rewards: { xp: 400, credits: 80 },
        requiredLevel: 1,
        timeLimit: 1800
      },
      
      // Daily missions
      {
        name: 'Daily Patrol',
        description: 'Complete a routine patrol of the area',
        type: 'daily',
        difficulty: 'easy',
        objectives: [
          { id: 1, description: 'Visit 3 checkpoints', required: true },
          { id: 2, description: 'Report any anomalies', required: false }
        ],
        rewards: { xp: 200, credits: 50 },
        requiredLevel: 1,
        timeLimit: null
      },
      {
        name: 'Combat Training',
        description: 'Complete combat training exercises',
        type: 'daily',
        difficulty: 'medium',
        objectives: [
          { id: 1, description: 'Defeat 10 training dummies', required: true },
          { id: 2, description: 'Achieve 80% accuracy', required: true }
        ],
        rewards: { xp: 300, credits: 75 },
        requiredLevel: 1,
        timeLimit: 900
      },
      
      // Weekly missions
      {
        name: 'Weekly Challenge: Endurance',
        description: 'Test your endurance in extended combat',
        type: 'weekly',
        difficulty: 'extreme',
        objectives: [
          { id: 1, description: 'Survive 10 waves of enemies', required: true },
          { id: 2, description: 'Maintain 50% health', required: false },
          { id: 3, description: 'No healing items', required: false }
        ],
        rewards: { xp: 5000, credits: 1000, items: ['Legendary Core'] },
        requiredLevel: 20,
        timeLimit: null
      }
    ];
    
    for (const mission of missionData) {
      await db.insert(missions).values(mission).onConflictDoNothing();
    }
    console.log(`✅ Created ${missionData.length} missions`);
    
    // Seed equipment
    console.log('Creating equipment...');
    const equipmentData = [
      // Weapons
      {
        name: 'Type-3 Sword',
        type: 'weapon',
        tier: 'common',
        stats: { attack: 50, speed: 1.2 },
        requirements: { level: 1 },
        description: 'Standard issue YoRHa blade'
      },
      {
        name: 'Type-40 Blade',
        type: 'weapon',
        tier: 'rare',
        stats: { attack: 120, speed: 1.0, critical: 10 },
        requirements: { level: 10 },
        description: 'Enhanced combat blade with improved balance'
      },
      {
        name: 'Virtuous Contract',
        type: 'weapon',
        tier: 'legendary',
        stats: { attack: 250, speed: 1.1, critical: 20, combo: 5 },
        requirements: { level: 25 },
        description: 'A white blade given to 2B by the organization'
      },
      
      // Armor
      {
        name: 'YoRHa Uniform',
        type: 'armor',
        tier: 'common',
        stats: { defense: 30, resistance: 10 },
        requirements: { level: 1 },
        description: 'Standard YoRHa combat uniform'
      },
      {
        name: 'Reinforced Plating',
        type: 'armor',
        tier: 'rare',
        stats: { defense: 80, resistance: 25, mobility: -5 },
        requirements: { level: 15 },
        description: 'Heavy armor plating for increased protection'
      },
      
      // Accessories
      {
        name: 'Scanner Module',
        type: 'accessory',
        tier: 'common',
        stats: { scan_range: 50, detection: 10 },
        requirements: { level: 1 },
        description: 'Basic scanning equipment'
      },
      {
        name: 'Speed Chip',
        type: 'accessory',
        tier: 'rare',
        stats: { movement_speed: 15, evasion: 10 },
        requirements: { level: 8 },
        description: 'Enhances movement and evasion capabilities'
      },
      {
        name: 'Critical Chip',
        type: 'accessory',
        tier: 'rare',
        stats: { critical_rate: 15, critical_damage: 30 },
        requirements: { level: 12 },
        description: 'Increases critical hit chance and damage'
      }
    ];
    
    for (const item of equipmentData) {
      await db.insert(equipment).values(item).onConflictDoNothing();
    }
    console.log(`✅ Created ${equipmentData.length} equipment items`);
    
    // Create demo users
    console.log('Creating demo users...');
    const demoUsers = [
      {
        unitId: '2B-0001',
        email: 'demo@yorha.net',
        passwordHash: await bcrypt.hash('demo123', 12),
        name: 'Unit 2B',
        unitType: 'combat' as const,
        bio: 'YoRHa No.2 Type B. Elite combat android. Fighting for the glory of mankind.',
        level: 15,
        xp: 2500,
        rank: 'B' as const,
        missionsCompleted: 27,
        combatRating: '85.5',
        hoursActive: 142,
        achievementsUnlocked: 8,
        emailVerified: true,
        emailVerificationToken: null
      },
      {
        unitId: '9S-0002',
        email: 'scanner@yorha.net',
        passwordHash: await bcrypt.hash('scan123', 12),
        name: 'Unit 9S',
        unitType: 'scanner' as const,
        bio: 'YoRHa No.9 Type S. Scanner model specializing in reconnaissance and hacking.',
        level: 12,
        xp: 1800,
        rank: 'C' as const,
        missionsCompleted: 19,
        combatRating: '72.3',
        hoursActive: 98,
        achievementsUnlocked: 6,
        emailVerified: true,
        emailVerificationToken: null
      },
      {
        unitId: 'A2-0003',
        email: 'attacker@yorha.net',
        passwordHash: await bcrypt.hash('attack123', 12),
        name: 'Unit A2',
        unitType: 'combat' as const,
        bio: 'YoRHa Type A No.2. Prototype attacker model. Status: Rogue.',
        level: 25,
        xp: 8750,
        rank: 'A' as const,
        missionsCompleted: 89,
        combatRating: '96.8',
        hoursActive: 521,
        achievementsUnlocked: 15,
        emailVerified: true,
        emailVerificationToken: null
      }
    ];
    
    for (const user of demoUsers) {
      await db.insert(units).values(user).onConflictDoNothing();
    }
    console.log(`✅ Created ${demoUsers.length} demo users`);
    
    console.log('====================================');
    console.log('✅ Database seed completed successfully!');
    console.log('Demo accounts:');
    console.log('  Email: demo@yorha.net | Password: demo123');
    console.log('  Email: scanner@yorha.net | Password: scan123');
    console.log('  Email: attacker@yorha.net | Password: attack123');
    console.log('====================================');
    console.log('FOR THE GLORY OF MANKIND');
    
  } catch (error) {
    console.error('❌ Seed failed:', error);
    process.exit(1);
  }
  
  process.exit(0);
}

// Run seed
seed();