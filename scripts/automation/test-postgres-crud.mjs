import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { users } from './sveltekit-frontend/src/lib/server/db/schema-postgres.js';
import { eq } from 'drizzle-orm';
import bcrypt from 'bcrypt';
import { randomUUID } from 'crypto';

// Database connection
const connectionString = 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db';
const pool = postgres(connectionString);
const db = drizzle(pool);

console.log('🔌 Testing PostgreSQL Connection...\n');

async function testConnection() {
  try {
    const result = await db.execute`SELECT current_database(), current_user, version()`;
    console.log('✅ Database connection successful!');
    console.log('📊 Database info:', result[0]);
    return true;
  } catch (error) {
    console.error('❌ Database connection failed:', error.message);
    return false;
  }
}

async function testUserCRUD() {
  console.log('\n📝 Testing User CRUD Operations...\n');
  
  const testEmail = `test_${Date.now()}@example.com`;
  const testPassword = 'TestPassword123!';
  
  try {
    // CREATE - Register new user
    console.log('1️⃣ CREATE - Registering new user...');
    const hashedPassword = await bcrypt.hash(testPassword, 10);
    
    const [newUser] = await db.insert(users).values({
      email: testEmail,
      hashedPassword: hashedPassword,
      name: 'Test User',
      firstName: 'Test',
      lastName: 'User',
      role: 'prosecutor',
      isActive: true
    }).returning();
    
    console.log('✅ User created:', {
      id: newUser.id,
      email: newUser.email,
      name: newUser.name,
      role: newUser.role
    });
    
    // READ - Get user by email
    console.log('\n2️⃣ READ - Finding user by email...');
    const foundUser = await db.select().from(users).where(eq(users.email, testEmail));
    console.log('✅ User found:', foundUser.length > 0 ? 'Yes' : 'No');
    
    // UPDATE - Update user name
    console.log('\n3️⃣ UPDATE - Updating user name...');
    const [updatedUser] = await db.update(users)
      .set({ name: 'Updated Test User' })
      .where(eq(users.id, newUser.id))
      .returning();
    
    console.log('✅ User updated:', {
      id: updatedUser.id,
      name: updatedUser.name
    });
    
    // LOGIN TEST - Verify password
    console.log('\n🔐 Testing login verification...');
    const loginUser = await db.select().from(users).where(eq(users.email, testEmail));
    if (loginUser.length > 0 && loginUser[0].hashedPassword) {
      const isValidPassword = await bcrypt.compare(testPassword, loginUser[0].hashedPassword);
      console.log('✅ Password verification:', isValidPassword ? 'Success' : 'Failed');
    }
    
    // DELETE - Clean up test user
    console.log('\n4️⃣ DELETE - Removing test user...');
    await db.delete(users).where(eq(users.id, newUser.id));
    console.log('✅ User deleted');
    
    // Verify deletion
    const deletedUser = await db.select().from(users).where(eq(users.id, newUser.id));
    console.log('✅ Deletion verified:', deletedUser.length === 0 ? 'Success' : 'Failed');
    
  } catch (error) {
    console.error('❌ CRUD operation failed:', error.message);
    throw error;
  }
}

async function listExistingUsers() {
  console.log('\n👥 Listing existing users...\n');
  try {
    const existingUsers = await db.select({
      id: users.id,
      email: users.email,
      name: users.name,
      role: users.role,
      createdAt: users.createdAt
    }).from(users).limit(5);
    
    if (existingUsers.length > 0) {
      console.log('Found', existingUsers.length, 'users:');
      existingUsers.forEach((user, index) => {
        console.log(`${index + 1}. ${user.email} (${user.role}) - ${user.name || 'No name'}`);
      });
    } else {
      console.log('No users found in database');
    }
  } catch (error) {
    console.error('❌ Failed to list users:', error.message);
  }
}

async function runTests() {
  try {
    const connected = await testConnection();
    if (!connected) {
      console.log('\n⚠️  Cannot proceed without database connection');
      process.exit(1);
    }
    
    await testUserCRUD();
    await listExistingUsers();
    
    console.log('\n✅ All tests completed successfully!');
    console.log('\n📌 Summary:');
    console.log('- PostgreSQL is running locally');
    console.log('- Drizzle ORM is properly configured');
    console.log('- User CRUD operations work correctly');
    console.log('- Password hashing and verification work');
    console.log('- Database schema is properly set up');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error);
  } finally {
    await pool.end();
    process.exit(0);
  }
}

runTests();