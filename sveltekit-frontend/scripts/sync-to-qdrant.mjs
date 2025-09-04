#!/usr/bin/env node
/**
 * Qdrant Synchronization Script
 * Syncs PostgreSQL vector data with Qdrant collections
 */

import { createQdrantService } from '../src/lib/server/db/qdrant-integration.js';

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  red: '\x1b[31m'
};

const log = {
  info: (msg) => console.log(`${colors.cyan}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✅${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}❌${colors.reset} ${msg}`),
  step: (msg) => console.log(`${colors.blue}🔄${colors.reset} ${msg}`),
};

async function main() {
  console.log(`${colors.bright}🚀 Qdrant Synchronization for Legal AI Platform${colors.reset}\n`);

  const batchSize = process.argv.includes('--batch-size') 
    ? parseInt(process.argv[process.argv.indexOf('--batch-size') + 1]) || 100 
    : 100;

  const dryRun = process.argv.includes('--dry-run');
  
  if (dryRun) {
    log.warning('Running in DRY RUN mode - no actual changes will be made');
  }

  let qdrantService;

  try {
    // Initialize service
    log.step('Initializing Qdrant-PostgreSQL service...');
    qdrantService = createQdrantService();
    log.success('Service initialized successfully');

    // Health check
    log.step('Performing health check...');
    const health = await qdrantService.healthCheck();
    
    console.log('\n📊 Health Check Results:');
    console.log(`   PostgreSQL: ${health.postgresql ? '✅' : '❌'} Connected`);
    console.log(`   Qdrant: ${health.qdrant ? '✅' : '❌'} Connected`);
    console.log(`   Collections: ${health.collections.join(', ')}`);
    console.log(`   Total Documents: ${health.syncStatus.totalDocuments}`);
    console.log(`   Synced Documents: ${health.syncStatus.syncedDocuments}`);
    console.log(`   Pending Syncs: ${health.syncStatus.pendingSyncs}`);

    if (!health.postgresql) {
      log.error('PostgreSQL is not accessible. Please check your database connection.');
      process.exit(1);
    }

    if (!health.qdrant) {
      log.warning('Qdrant is not accessible. Sync will be limited to PostgreSQL operations.');
    }

    // Ensure collections exist
    if (health.qdrant) {
      log.step('Ensuring Qdrant collections exist...');
      await qdrantService.ensureCollection('legal_documents', 384, 'Cosine');
      await qdrantService.ensureCollection('cases', 384, 'Cosine');
      await qdrantService.ensureCollection('users', 384, 'Cosine');
      log.success('Collections verified/created');
    }

    // Batch sync
    if (health.syncStatus.pendingSyncs > 0 && !dryRun) {
      log.step(`Starting batch sync of ${health.syncStatus.pendingSyncs} documents (batch size: ${batchSize})...`);
      
      const syncResults = await qdrantService.batchSyncToQdrant('document', batchSize);
      
      console.log('\n📋 Sync Results:');
      console.log(`   ✅ Synced: ${syncResults.synced} documents`);
      console.log(`   ❌ Failed: ${syncResults.failed} documents`);
      
      if (syncResults.errors.length > 0) {
        console.log('\n⚠️ Errors encountered:');
        syncResults.errors.forEach(error => console.log(`   • ${error}`));
      }

      if (syncResults.synced > 0) {
        log.success(`Successfully synced ${syncResults.synced} documents to Qdrant`);
      }
    } else if (dryRun) {
      log.info(`DRY RUN: Would sync ${health.syncStatus.pendingSyncs} documents`);
    } else {
      log.info('No documents need syncing');
    }

    // Final health check
    log.step('Performing final health check...');
    const finalHealth = await qdrantService.healthCheck();
    
    console.log('\n🎯 Final Status:');
    console.log(`   Total Documents: ${finalHealth.syncStatus.totalDocuments}`);
    console.log(`   Synced Documents: ${finalHealth.syncStatus.syncedDocuments}`);
    console.log(`   Pending Syncs: ${finalHealth.syncStatus.pendingSyncs}`);
    console.log(`   Sync Rate: ${((finalHealth.syncStatus.syncedDocuments / finalHealth.syncStatus.totalDocuments) * 100).toFixed(1)}%`);

  } catch (error) {
    log.error(`Sync operation failed: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  } finally {
    if (qdrantService) {
      await qdrantService.close();
    }
  }

  console.log(`\n${colors.green}${colors.bright}✅ Qdrant sync completed successfully!${colors.reset}`);
  
  console.log('\n🎯 Usage examples:');
  console.log('   npm run qdrant:sync                    # Sync all pending documents');
  console.log('   npm run qdrant:sync -- --batch-size 50 # Use smaller batch size');
  console.log('   npm run qdrant:sync -- --dry-run       # Preview what would be synced');
  console.log('   npm run qdrant:health                  # Check Qdrant status');
}

// Handle errors
process.on('uncaughtException', (error) => {
  console.error(`${colors.red}❌ Uncaught exception: ${error.message}${colors.reset}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(`${colors.red}❌ Unhandled rejection: ${reason}${colors.reset}`);
  process.exit(1);
});

// Run main function
main().catch((error) => {
  console.error(`${colors.red}❌ Sync failed: ${error.message}${colors.reset}`);
  process.exit(1);
});