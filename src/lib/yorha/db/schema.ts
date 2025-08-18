// Drizzle ORM Schema for YoRHa Interface
import { 
  pgTable, 
  serial, 
  text, 
  varchar, 
  timestamp, 
  integer, 
  boolean, 
  jsonb,
  uuid,
  decimal,
  index,
  uniqueIndex,
  pgEnum
} from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

// Enums
export const unitTypeEnum = pgEnum('unit_type', ['combat', 'scanner', 'support', 'operator', 'healer']);
export const rankEnum = pgEnum('rank', ['E', 'D', 'C', 'B', 'A', 'S', 'SS']);
export const activityTypeEnum = pgEnum('activity_type', [
  'login', 
  'logout', 
  'mission_start', 
  'mission_complete', 
  'level_up', 
  'achievement_unlock',
  'equipment_change',
  'profile_update',
  'combat_action',
  'system_sync'
]);

// Users/Units Table
export const units = pgTable('units', {
  id: uuid('id').primaryKey().defaultRandom(),
  unitId: varchar('unit_id', { length: 20 }).unique().notNull(), // e.g., "2B-4827"
  email: varchar('email', { length: 255 }).unique().notNull(),
  passwordHash: text('password_hash').notNull(),
  
  // Profile Information
  name: varchar('name', { length: 100 }).notNull(),
  unitType: unitTypeEnum('unit_type').default('combat').notNull(),
  bio: text('bio').default('YoRHa Unit. Fighting for the glory of mankind.'),
  avatarUrl: text('avatar_url'),
  
  // Level & Experience
  level: integer('level').default(1).notNull(),
  xp: integer('xp').default(0).notNull(),
  rank: rankEnum('rank').default('E').notNull(),
  
  // Stats
  missionsCompleted: integer('missions_completed').default(0).notNull(),
  combatRating: decimal('combat_rating', { precision: 5, scale: 2 }).default('50.00').notNull(),
  hoursActive: integer('hours_active').default(0).notNull(),
  achievementsUnlocked: integer('achievements_unlocked').default(0).notNull(),
  
  // Vector embedding for AI features (Nomic dimensions) - TODO: Re-enable when pgvector is properly configured
  // embedding: vector('embedding', { dimensions: 768 }),
  
  // Security
  twoFactorEnabled: boolean('two_factor_enabled').default(false).notNull(),
  twoFactorSecret: text('two_factor_secret'),
  emailVerified: boolean('email_verified').default(false).notNull(),
  emailVerificationToken: varchar('email_verification_token', { length: 255 }),
  
  // Settings
  settings: jsonb('settings').default({
    notifications: true,
    profileVisibility: 'public',
    showActivityStatus: true,
    dataCollection: false,
    theme: 'yorha-default'
  }).notNull(),
  
  // Timestamps
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
  lastLoginAt: timestamp('last_login_at'),
  deletedAt: timestamp('deleted_at')
}, (table) => ({
  emailIdx: index('units_email_idx').on(table.email),
  unitIdIdx: uniqueIndex('units_unit_id_idx').on(table.unitId),
  levelIdx: index('units_level_idx').on(table.level),
  // embeddingIdx: index('units_embedding_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

// Sessions Table
export const sessions = pgTable('sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => units.id, { onDelete: 'cascade' }).notNull(),
  token: text('token').unique().notNull(),
  
  userAgent: text('user_agent'),
  ipAddress: varchar('ip_address', { length: 45 }),
  
  expiresAt: timestamp('expires_at').notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  lastActivityAt: timestamp('last_activity_at').defaultNow().notNull()
}, (table) => ({
  tokenIdx: uniqueIndex('sessions_token_idx').on(table.token),
  userIdIdx: index('sessions_user_id_idx').on(table.userId),
  expiresAtIdx: index('sessions_expires_at_idx').on(table.expiresAt)
}));

// User Activity Table
export const userActivity = pgTable('user_activity', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => units.id, { onDelete: 'cascade' }).notNull(),
  
  activityType: activityTypeEnum('activity_type').notNull(),
  description: text('description').notNull(),
  metadata: jsonb('metadata').default({}).notNull(),
  
  // Location/Context
  ipAddress: varchar('ip_address', { length: 45 }),
  userAgent: text('user_agent'),
  sessionId: uuid('session_id').references(() => sessions.id, { onDelete: 'set null' }),
  
  createdAt: timestamp('created_at').defaultNow().notNull()
}, (table) => ({
  userIdIdx: index('activity_user_id_idx').on(table.userId),
  typeIdx: index('activity_type_idx').on(table.activityType),
  createdAtIdx: index('activity_created_at_idx').on(table.createdAt)
}));

// Achievements Table
export const achievements = pgTable('achievements', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  name: varchar('name', { length: 100 }).notNull(),
  description: text('description').notNull(),
  icon: varchar('icon', { length: 50 }).notNull(), // Icon component name
  category: varchar('category', { length: 50 }).notNull(),
  
  requiredValue: integer('required_value').default(1).notNull(),
  xpReward: integer('xp_reward').default(100).notNull(),
  
  hidden: boolean('hidden').default(false).notNull(),
  active: boolean('active').default(true).notNull(),
  
  createdAt: timestamp('created_at').defaultNow().notNull()
}, (table) => ({
  categoryIdx: index('achievements_category_idx').on(table.category),
  activeIdx: index('achievements_active_idx').on(table.active)
}));

// User Achievements Table (Junction)
export const userAchievements = pgTable('user_achievements', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => units.id, { onDelete: 'cascade' }).notNull(),
  achievementId: uuid('achievement_id').references(() => achievements.id, { onDelete: 'cascade' }).notNull(),
  
  progress: integer('progress').default(0).notNull(),
  unlockedAt: timestamp('unlocked_at'),
  
  createdAt: timestamp('created_at').defaultNow().notNull()
}, (table) => ({
  userAchievementIdx: uniqueIndex('user_achievement_idx').on(table.userId, table.achievementId)
}));

// Equipment Table
export const equipment = pgTable('equipment', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  name: varchar('name', { length: 100 }).notNull(),
  type: varchar('type', { length: 50 }).notNull(), // weapon, armor, accessory
  tier: varchar('tier', { length: 20 }).notNull(), // common, rare, legendary
  
  stats: jsonb('stats').default({}).notNull(),
  requirements: jsonb('requirements').default({}).notNull(),
  
  description: text('description'),
  iconUrl: text('icon_url'),
  
  createdAt: timestamp('created_at').defaultNow().notNull()
});

// User Equipment Table
export const userEquipment = pgTable('user_equipment', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => units.id, { onDelete: 'cascade' }).notNull(),
  equipmentId: uuid('equipment_id').references(() => equipment.id, { onDelete: 'cascade' }).notNull(),
  
  level: integer('level').default(1).notNull(),
  equipped: boolean('equipped').default(false).notNull(),
  slot: varchar('slot', { length: 50 }), // main_weapon, sub_weapon, etc.
  
  enhancements: jsonb('enhancements').default({}).notNull(),
  
  acquiredAt: timestamp('acquired_at').defaultNow().notNull()
}, (table) => ({
  userEquipmentIdx: index('user_equipment_idx').on(table.userId, table.equipped)
}));

// Export types
export type Unit = typeof units.$inferSelect;
export type NewUnit = typeof units.$inferInsert;
export type Session = typeof sessions.$inferSelect;
export type NewSession = typeof sessions.$inferInsert;
export type UserActivity = typeof userActivity.$inferSelect;
export type NewUserActivity = typeof userActivity.$inferInsert;

// Missions Table
export const missions = pgTable('missions', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  name: varchar('name', { length: 200 }).notNull(),
  description: text('description').notNull(),
  type: varchar('type', { length: 50 }).notNull(), // main, side, daily, weekly
  difficulty: varchar('difficulty', { length: 20 }).notNull(), // easy, medium, hard, extreme
  
  objectives: jsonb('objectives').default([]).notNull(),
  rewards: jsonb('rewards').default({}).notNull(),
  
  requiredLevel: integer('required_level').default(1).notNull(),
  timeLimit: integer('time_limit'), // in seconds
  
  active: boolean('active').default(true).notNull(),
  
  createdAt: timestamp('created_at').defaultNow().notNull()
}, (table) => ({
  typeIdx: index('missions_type_idx').on(table.type),
  activeIdx: index('missions_active_idx').on(table.active)
}));

// User Missions Table
export const userMissions = pgTable('user_missions', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => units.id, { onDelete: 'cascade' }).notNull(),
  missionId: uuid('mission_id').references(() => missions.id, { onDelete: 'cascade' }).notNull(),
  
  status: varchar('status', { length: 20 }).default('active').notNull(), // active, completed, failed, abandoned
  progress: jsonb('progress').default({}).notNull(),
  
  startedAt: timestamp('started_at').defaultNow().notNull(),
  completedAt: timestamp('completed_at'),
  
  score: integer('score'),
  timeTaken: integer('time_taken') // in seconds
}, (table) => ({
  userMissionIdx: index('user_mission_idx').on(table.userId, table.status)
}));

// Password Reset Tokens
export const passwordResetTokens = pgTable('password_reset_tokens', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => units.id, { onDelete: 'cascade' }).notNull(),
  token: text('token').unique().notNull(),
  
  expiresAt: timestamp('expires_at').notNull(),
  usedAt: timestamp('used_at'),
  createdAt: timestamp('created_at').defaultNow().notNull()
}, (table) => ({
  tokenIdx: uniqueIndex('reset_token_idx').on(table.token),
  userIdIdx: index('reset_user_id_idx').on(table.userId)
}));

// Documents Table for RAG
export const documents = pgTable('documents', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content').notNull(),
  source: varchar('source', { length: 100 }).notNull(), // url, upload, manual, etc.
  
  userId: uuid('user_id').references(() => units.id, { onDelete: 'set null' }),
  
  metadata: jsonb('metadata').default({}).notNull(),
  tags: text('tags').array(),
  
  // Vector embedding for document (Nomic dimensions) - TODO: Re-enable when pgvector is properly configured
  // embedding: vector('embedding', { dimensions: 768 }),
  
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull()
}, (table) => ({
  titleIdx: index('documents_title_idx').on(table.title),
  sourceIdx: index('documents_source_idx').on(table.source),
  userIdIdx: index('documents_user_id_idx').on(table.userId),
  // embeddingIdx: index('documents_embedding_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

// Document Chunks Table for RAG
export const documentChunks = pgTable('document_chunks', {
  id: uuid('id').primaryKey().defaultRandom(),
  documentId: uuid('document_id').references(() => documents.id, { onDelete: 'cascade' }).notNull(),
  
  chunkIndex: integer('chunk_index').notNull(),
  content: text('content').notNull(),
  
  metadata: jsonb('metadata').default({}).notNull(),
  
  // Vector embedding for chunk (Nomic dimensions) - TODO: Re-enable when pgvector is properly configured
  // embedding: vector('embedding', { dimensions: 768 }),
  
  createdAt: timestamp('created_at').defaultNow().notNull()
}, (table) => ({
  documentIdx: index('chunks_document_idx').on(table.documentId),
  chunkIdx: uniqueIndex('chunks_doc_index_idx').on(table.documentId, table.chunkIndex),
  // embeddingIdx: index('chunks_embedding_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

// Conversations Table for Chat History
export const conversations = pgTable('conversations', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => units.id, { onDelete: 'cascade' }).notNull(),
  
  title: varchar('title', { length: 255 }),
  messages: jsonb('messages').default([]).notNull(), // Array of {role, content, timestamp}
  
  metadata: jsonb('metadata').default({}).notNull(),
  
  // Summary embedding for conversation (Nomic dimensions) - TODO: Re-enable when pgvector is properly configured
  // embedding: vector('embedding', { dimensions: 768 }),
  
  lastMessageAt: timestamp('last_message_at').defaultNow().notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull()
}, (table) => ({
  userIdIdx: index('conversations_user_id_idx').on(table.userId),
  lastMessageIdx: index('conversations_last_message_idx').on(table.lastMessageAt),
  // embeddingIdx: index('conversations_embedding_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

// Knowledge Base Table
export const knowledgeBase = pgTable('knowledge_base', {
  id: uuid('id').primaryKey().defaultRandom(),
  
  category: varchar('category', { length: 100 }).notNull(),
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content').notNull(),
  
  tags: text('tags').array(),
  metadata: jsonb('metadata').default({}).notNull(),
  
  // Vector embedding (Nomic dimensions) - TODO: Re-enable when pgvector is properly configured
  // embedding: vector('embedding', { dimensions: 768 }),
  
  views: integer('views').default(0).notNull(),
  helpful: integer('helpful').default(0).notNull(),
  
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull()
}, (table) => ({
  categoryIdx: index('kb_category_idx').on(table.category),
  titleIdx: index('kb_title_idx').on(table.title),
  // embeddingIdx: index('kb_embedding_idx').using('hnsw', table.embedding.op('vector_cosine_ops'))
}));

// Export additional types
export type Achievement = typeof achievements.$inferSelect;
export type UserAchievement = typeof userAchievements.$inferSelect;
export type Equipment = typeof equipment.$inferSelect;
export type UserEquipment = typeof userEquipment.$inferSelect;
export type Mission = typeof missions.$inferSelect;
export type UserMission = typeof userMissions.$inferSelect;
export type Document = typeof documents.$inferSelect;
export type NewDocument = typeof documents.$inferInsert;
export type DocumentChunk = typeof documentChunks.$inferSelect;
export type Conversation = typeof conversations.$inferSelect;
export type KnowledgeBase = typeof knowledgeBase.$inferSelect;