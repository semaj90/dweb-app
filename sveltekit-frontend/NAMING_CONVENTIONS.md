# 🔄 Naming Conventions & Case Transformation Guide

## Overview

This project follows a **dual naming convention** system to maintain consistency with industry standards:

- **Database (PostgreSQL + Drizzle)**: `snake_case` - PostgreSQL standard
- **Frontend (SvelteKit + TypeScript)**: `camelCase` - JavaScript standard

## 📋 Quick Reference

### Database Schema (`snake_case`)
```sql
-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  first_name VARCHAR(100),
  last_name VARCHAR(100), 
  hashed_password VARCHAR(255),
  is_active BOOLEAN DEFAULT true,
  email_verified BOOLEAN DEFAULT false,
  avatar_url VARCHAR(500),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Frontend Types (`camelCase`)
```typescript
interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  hashedPassword?: string;
  isActive: boolean;
  emailVerified: boolean;
  avatarUrl?: string;
  createdAt: Date;
  updatedAt: Date;
}
```

## 🛠️ Implementation

### 1. Use the Transformation Utilities

Import the case transformation helpers:

```typescript
import { 
  toCamelCase, 
  toSnakeCase, 
  transformUserForFrontend,
  transformUserForDatabase,
  apiResponse,
  dbQuery 
} from '$lib/utils/case-transform';
```

### 2. API Server Functions

**✅ Correct Implementation:**

```typescript
// src/routes/api/users/+server.ts
export const GET: RequestHandler = async ({ locals }) => {
  // 1. Query database with snake_case field names
  const users = await db.query.users.findMany({
    columns: {
      id: true,
      email: true,
      first_name: true,        // Database field
      last_name: true,         // Database field  
      avatar_url: true,        // Database field
      created_at: true,        // Database field
      is_active: true          // Database field
    }
  });
  
  // 2. Transform to camelCase for frontend
  const frontendUsers = users.map(transformUserForFrontend);
  
  return json({ users: frontendUsers });
};

export const POST: RequestHandler = async ({ request }) => {
  // 1. Receive camelCase data from frontend
  const frontendData = await request.json();
  
  // 2. Transform to snake_case for database
  const dbData = transformUserForDatabase(frontendData);
  
  // 3. Insert using snake_case field names
  const [newUser] = await db.insert(users).values({
    email: dbData.email,
    first_name: dbData.first_name,
    last_name: dbData.last_name,
    // ... other snake_case fields
  }).returning();
  
  // 4. Transform result back to camelCase
  const frontendUser = transformUserForFrontend(newUser);
  
  return json({ user: frontendUser });
};
```

**❌ Incorrect Implementation:**

```typescript
// DON'T DO THIS - mixing camelCase with database queries
const user = await db.query.users.findFirst({
  columns: {
    firstName: true,    // ❌ Database doesn't have this field
    lastName: true,     // ❌ Database doesn't have this field
    avatarUrl: true,    // ❌ Database doesn't have this field
  }
});

// DON'T DO THIS - returning snake_case to frontend
return json({ 
  user: {
    first_name: user.first_name,  // ❌ Frontend expects camelCase
    avatar_url: user.avatar_url   // ❌ Frontend expects camelCase
  }
});
```

### 3. Drizzle Schema Definitions

Always use snake_case in schema definitions:

```typescript
// src/lib/server/db/schema-postgres.ts
export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).notNull(),
  first_name: varchar('first_name', { length: 100 }),      // ✅ snake_case
  last_name: varchar('last_name', { length: 100 }),        // ✅ snake_case
  avatar_url: varchar('avatar_url', { length: 500 }),      // ✅ snake_case
  email_verified: boolean('email_verified').default(false), // ✅ snake_case
  created_at: timestamp('created_at').defaultNow(),        // ✅ snake_case
  updated_at: timestamp('updated_at').defaultNow()         // ✅ snake_case
});
```

### 4. Frontend Component Usage

Components should only deal with camelCase:

```svelte
<!-- UserProfile.svelte -->
<script lang="ts">
  interface User {
    firstName?: string;    // ✅ camelCase
    lastName?: string;     // ✅ camelCase  
    avatarUrl?: string;    // ✅ camelCase
    emailVerified: boolean; // ✅ camelCase
  }
  
  let { user }: { user: User } = $props();
</script>

<div class="profile">
  <img src={user.avatarUrl} alt="Avatar" />
  <h2>{user.firstName} {user.lastName}</h2>
  {#if user.emailVerified}
    <span class="verified">✓ Verified</span>
  {/if}
</div>
```

## 🎯 Field Mappings

Common field transformations handled automatically:

| Database (`snake_case`) | Frontend (`camelCase`) |
|------------------------|------------------------|
| `first_name` | `firstName` |
| `last_name` | `lastName` |
| `email_verified` | `emailVerified` |
| `is_active` | `isActive` |
| `avatar_url` | `avatarUrl` |
| `created_at` | `createdAt` |
| `updated_at` | `updatedAt` |
| `case_number` | `caseNumber` |
| `file_path` | `filePath` |
| `hash_sha256` | `hashSha256` |

## 🚀 Advanced Usage

### Batch Transformations

```typescript
// Transform arrays
const frontendUsers = transformArray(dbUsers, transformUserForFrontend);

// Generic transformations
const camelCaseData = toCamelCase<MyFrontendType>(snakeCaseData);
const snakeCaseData = toSnakeCase<MyDatabaseType>(camelCaseData);
```

### Query Helpers

```typescript
// Convert camelCase input to snake_case for database queries
const dbFilters = dbQuery({ firstName: 'John', isActive: true });
// Result: { first_name: 'John', is_active: true }

// Convert snake_case results to camelCase for API response
const response = apiResponse<User[]>(dbResults);
```

## 🔧 Development Workflow

1. **Database Migrations**: Always use `snake_case`
2. **Schema Definitions**: Always use `snake_case`
3. **API Endpoints**: Transform at the boundary (request/response)
4. **Frontend Types**: Always use `camelCase`
5. **Component Props**: Always use `camelCase`

## ✅ Testing

Test both transformations work correctly:

```typescript
import { toCamelCase, toSnakeCase } from '$lib/utils/case-transform';

test('snake_case to camelCase transformation', () => {
  const input = { first_name: 'John', is_active: true };
  const result = toCamelCase(input);
  expect(result).toEqual({ firstName: 'John', isActive: true });
});

test('camelCase to snake_case transformation', () => {
  const input = { firstName: 'John', isActive: true };
  const result = toSnakeCase(input);
  expect(result).toEqual({ first_name: 'John', is_active: true });
});
```

## 🎯 Benefits

1. **PostgreSQL Standard Compliance**: Database follows SQL naming conventions
2. **JavaScript Best Practices**: Frontend follows JS/TS conventions  
3. **Clear Boundaries**: Transformations happen at API boundaries
4. **Type Safety**: TypeScript catches naming mismatches
5. **Consistency**: Automated transformations prevent manual errors
6. **Maintainability**: Single source of truth for field mappings

## 🚨 Common Mistakes to Avoid

1. **Don't mix naming conventions** within the same layer
2. **Don't transform inside components** - do it at API boundaries
3. **Don't hardcode field mappings** - use the transformation utilities
4. **Don't forget to transform both directions** (request + response)
5. **Don't bypass transformations** for "simple" cases

---

**Remember**: Database speaks `snake_case`, Frontend speaks `camelCase`, transformations happen at the API boundary! 🔄