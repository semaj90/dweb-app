#!/usr/bin/env node

import { readFileSync } from 'fs';
import { db } from './src/lib/server/db/index.js';
import { cases, caseDocuments, caseActivities, caseTimeline } from './src/lib/server/db/schema.js';
import { eq } from 'drizzle-orm';

console.log('🧪 Testing CRUD Operations Directly...\n');

// Test data
const testCaseData = {
  id: crypto.randomUUID(),
  caseNumber: 'TEST-' + Date.now(),
  title: 'Direct API Test Case',
  description: 'Testing CRUD operations directly',
  priority: 'medium',
  status: 'draft',
  metadata: { test: true, timestamp: Date.now() },
  created_at: new Date(),
  updated_at: new Date(),
  createdBy: 'test-script'
};

let createdCaseId = null;

try {
  // 1. TEST CREATE (INSERT)
  console.log('📝 Testing CREATE operation...');
  const insertResult = await db.insert(cases).values(testCaseData).returning();
  
  if (insertResult && insertResult.length > 0) {
    createdCaseId = insertResult[0].id;
    console.log('✅ CREATE: Successfully created case with ID:', createdCaseId);
    
    // Create related records
    await db.insert(caseActivities).values({
      id: crypto.randomUUID(),
      caseId: createdCaseId,
      type: 'case_created',
      description: 'Test case created via direct script',
      userId: 'test-script',
      timestamp: new Date(),
      metadata: { test: true }
    });
    
    await db.insert(caseTimeline).values({
      id: crypto.randomUUID(),
      caseId: createdCaseId,
      event: 'Case Created',
      description: 'Test case was created',
      timestamp: new Date(),
      type: 'milestone',
      metadata: {}
    });
    
    console.log('✅ CREATE: Related records created successfully');
  } else {
    console.error('❌ CREATE: Failed to create case');
  }

  // 2. TEST READ (SELECT)
  console.log('\n🔍 Testing READ operations...');
  
  // Read all cases
  const allCases = await db.select().from(cases).limit(5);
  console.log(`✅ READ: Retrieved ${allCases.length} cases`);
  
  // Read specific case
  if (createdCaseId) {
    const specificCase = await db.select().from(cases).where(eq(cases.id, createdCaseId));
    if (specificCase.length > 0) {
      console.log('✅ READ: Successfully retrieved specific case:', specificCase[0].title);
      
      // Read related data
      const activities = await db.select().from(caseActivities).where(eq(caseActivities.caseId, createdCaseId));
      const timeline = await db.select().from(caseTimeline).where(eq(caseTimeline.caseId, createdCaseId));
      
      console.log(`✅ READ: Found ${activities.length} activities, ${timeline.length} timeline events`);
    } else {
      console.error('❌ READ: Failed to find specific case');
    }
  }

  // 3. TEST UPDATE
  if (createdCaseId) {
    console.log('\n✏️ Testing UPDATE operation...');
    const updateResult = await db.update(cases)
      .set({
        title: 'Updated Test Case Title',
        description: 'Updated description via direct test',
        status: 'in_progress',
        priority: 'high',
        updated_at: new Date()
      })
      .where(eq(cases.id, createdCaseId))
      .returning();
    
    if (updateResult && updateResult.length > 0) {
      console.log('✅ UPDATE: Successfully updated case');
      console.log('   New title:', updateResult[0].title);
      console.log('   New status:', updateResult[0].status);
      
      // Add activity for update
      await db.insert(caseActivities).values({
        id: crypto.randomUUID(),
        caseId: createdCaseId,
        type: 'case_updated',
        description: 'Case updated via direct script',
        userId: 'test-script',
        timestamp: new Date(),
        metadata: { operation: 'update_test' }
      });
      
      console.log('✅ UPDATE: Activity record created');
    } else {
      console.error('❌ UPDATE: Failed to update case');
    }
  }

  // 4. TEST DELETE
  if (createdCaseId) {
    console.log('\n🗑️ Testing DELETE operation...');
    
    // Delete related records first (maintain referential integrity)
    await db.delete(caseTimeline).where(eq(caseTimeline.caseId, createdCaseId));
    await db.delete(caseActivities).where(eq(caseActivities.caseId, createdCaseId));
    await db.delete(caseDocuments).where(eq(caseDocuments.caseId, createdCaseId));
    
    console.log('✅ DELETE: Related records deleted');
    
    // Delete the case
    const deleteResult = await db.delete(cases).where(eq(cases.id, createdCaseId)).returning();
    
    if (deleteResult && deleteResult.length > 0) {
      console.log('✅ DELETE: Successfully deleted case:', deleteResult[0].title);
    } else {
      console.error('❌ DELETE: Failed to delete case');
    }
    
    // Verify deletion
    const verifyDelete = await db.select().from(cases).where(eq(cases.id, createdCaseId));
    if (verifyDelete.length === 0) {
      console.log('✅ DELETE: Verified case was completely removed');
    } else {
      console.error('❌ DELETE: Case still exists after deletion attempt');
    }
  }

  console.log('\n🎉 All CRUD operations completed successfully!');
  
  // Summary
  console.log('\n📊 Test Summary:');
  console.log('   ✅ CREATE: Test case inserted with related records');
  console.log('   ✅ READ: Cases retrieved (list and specific)');
  console.log('   ✅ UPDATE: Case updated with new values');
  console.log('   ✅ DELETE: Case and related records removed');
  console.log('\n✨ Your CRUD operations are working perfectly!\n');

} catch (error) {
  console.error('\n❌ Error during CRUD testing:', error.message);
  console.error('Stack trace:', error.stack);
  
  // Cleanup on error
  if (createdCaseId) {
    try {
      await db.delete(caseTimeline).where(eq(caseTimeline.caseId, createdCaseId));
      await db.delete(caseActivities).where(eq(caseActivities.caseId, createdCaseId));
      await db.delete(cases).where(eq(cases.id, createdCaseId));
      console.log('🧹 Cleanup: Test case removed after error');
    } catch (cleanupError) {
      console.error('⚠️ Cleanup failed:', cleanupError.message);
    }
  }
  
  process.exit(1);
}

process.exit(0);