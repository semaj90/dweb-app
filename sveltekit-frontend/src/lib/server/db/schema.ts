import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';

export const users = sqliteTable('users', {
  id: text('id').primaryKey(),
  email: text('email').notNull().unique(),
  firstName: text('first_name').notNull(),
  lastName: text('last_name').notNull(),
  role: text('role').default('user'),
  isActive: integer('is_active', { mode: 'boolean' }).default(true),
  emailVerified: integer('email_verified', { mode: 'boolean' }).default(false),
  createdAt: text('created_at').default('CURRENT_TIMESTAMP'),
  updatedAt: text('updated_at').default('CURRENT_TIMESTAMP')
});

export const sessions = sqliteTable('sessions', {
  id: text('id').primaryKey(),
  userId: text('user_id').references(() => users.id),
  expiresAt: text('expires_at').notNull()
});

export const evidence = sqliteTable('evidence', {
  id: text('id').primaryKey(),
  userId: text('user_id').references(() => users.id),
  filename: text('filename').notNull(),
  content: text('content'),
  metadata: text('metadata'),
  createdAt: text('created_at').default('CURRENT_TIMESTAMP')
});
