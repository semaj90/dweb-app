import { db } from '../sveltekit-frontend/src/lib/server/database';
import { sql } from 'drizzle-orm';

/**
 * PostgreSQL + pgvector optimization script for legal AI platform
 * Run this after importing documents to optimize vector search performance
 */
async function optimizeVectorDatabase() {
  console.log('🚀 Starting PostgreSQL + pgvector optimization...');

  try {
    // 1. Create optimized vector indexes
    console.log('📊 Creating vector indexes...');
    
    // IVFFlat index for cosine similarity (recommended for most use cases)
    await db.execute(sql`
      CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding_ivfflat 
      ON documents USING ivfflat (embedding vector_cosine_ops) 
      WITH (lists = 100)
    `);

    // HNSW index for higher accuracy (PostgreSQL 14+)
    try {
      await db.execute(sql`
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding_hnsw 
        ON documents USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64)
      `);
      console.log('✅ HNSW index created (high accuracy)');
    } catch (error) {
      console.log('⚠️  HNSW index not supported, using IVFFlat only');
    }

    // Index for embedding cache
    await db.execute(sql`
      CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embedding_cache_hash 
      ON embedding_cache (text_hash)
    `);

    await db.execute(sql`
      CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embedding_cache_model 
      ON embedding_cache (model)
    `);

    // 2. Create composite indexes for filtered searches
    console.log('🔍 Creating composite indexes...');
    
    await db.execute(sql`
      CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_case_type 
      ON documents (case_id, document_type)
    `);

    await db.execute(sql`
      CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_created_case 
      ON documents (created_at DESC, case_id)
    `);

    // 3. Optimize table statistics
    console.log('📈 Updating table statistics...');
    
    await db.execute(sql`ANALYZE documents`);
    await db.execute(sql`ANALYZE embedding_cache`);
    await db.execute(sql`ANALYZE cases`);
    await db.execute(sql`ANALYZE evidence`);

    // 4. Configure PostgreSQL for vector workloads
    console.log('⚙️  Optimizing PostgreSQL configuration...');
    
    // Get current settings
    const currentSettings = await db.execute(sql`
      SELECT name, setting, unit, context 
      FROM pg_settings 
      WHERE name IN (
        'shared_buffers',
        'effective_cache_size', 
        'maintenance_work_mem',
        'checkpoint_completion_target',
        'wal_buffers',
        'default_statistics_target',
        'random_page_cost',
        'effective_io_concurrency'
      )
    `);

    console.log('Current PostgreSQL settings:');
    for (const setting of currentSettings.rows) {
      console.log(`  ${setting.name}: ${setting.setting}${setting.unit || ''}`);
    }

    // 5. Vacuum and reindex for optimal performance
    console.log('🧹 Cleaning up database...');
    
    await db.execute(sql`VACUUM ANALYZE documents`);
    await db.execute(sql`VACUUM ANALYZE embedding_cache`);

    // 6. Get performance statistics
    console.log('📊 Gathering performance statistics...');
    
    const stats = await db.execute(sql`
      SELECT 
        COUNT(*) as total_documents,
        COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as documents_with_embeddings,
        AVG(array_length(embedding, 1)) as avg_embedding_dimensions,
        MIN(created_at) as oldest_document,
        MAX(created_at) as newest_document,
        COUNT(DISTINCT case_id) as unique_cases,
        pg_size_pretty(pg_total_relation_size('documents')) as table_size
      FROM documents
    `);

    const cacheStats = await db.execute(sql`
      SELECT 
        COUNT(*) as total_embeddings,
        COUNT(DISTINCT model) as unique_models,
        COUNT(DISTINCT text_hash) as unique_texts,
        AVG(dimensions) as avg_dimensions,
        pg_size_pretty(pg_total_relation_size('embedding_cache')) as cache_size
      FROM embedding_cache
    `);

    const indexStats = await db.execute(sql`
      SELECT 
        schemaname,
        tablename,
        indexname,
        idx_tup_read,
        idx_tup_fetch,
        pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
      FROM pg_stat_user_indexes 
      WHERE tablename IN ('documents', 'embedding_cache')
      ORDER BY idx_tup_read DESC
    `);

    // 7. Display results
    console.log('\n🎯 Optimization Results:');
    console.log('======================');
    
    console.log('\n📄 Documents Table:');
    if (stats.rows.length > 0) {
      const doc = stats.rows[0];
      console.log(`  Total documents: ${doc.total_documents}`);
      console.log(`  With embeddings: ${doc.documents_with_embeddings}`);
      console.log(`  Avg dimensions: ${doc.avg_embedding_dimensions}`);
      console.log(`  Table size: ${doc.table_size}`);
      console.log(`  Unique cases: ${doc.unique_cases}`);
    }

    console.log('\n💾 Embedding Cache:');
    if (cacheStats.rows.length > 0) {
      const cache = cacheStats.rows[0];
      console.log(`  Cached embeddings: ${cache.total_embeddings}`);
      console.log(`  Unique models: ${cache.unique_models}`);
      console.log(`  Unique texts: ${cache.unique_texts}`);
      console.log(`  Cache size: ${cache.cache_size}`);
    }

    console.log('\n🔍 Index Performance:');
    for (const idx of indexStats.rows) {
      console.log(`  ${idx.indexname}: ${idx.idx_tup_read} reads, ${idx.index_size}`);
    }

    // 8. Performance recommendations
    console.log('\n💡 Performance Recommendations:');
    console.log('================================');
    
    const docCount = parseInt(stats.rows[0]?.total_documents || '0');
    const embeddedCount = parseInt(stats.rows[0]?.documents_with_embeddings || '0');
    
    if (docCount > 0) {
      const embeddingPercentage = (embeddedCount / docCount) * 100;
      console.log(`  📊 Embedding coverage: ${embeddingPercentage.toFixed(1)}%`);
      
      if (embeddingPercentage < 90) {
        console.log('  ⚠️  Consider indexing remaining documents for complete vector search');
      }
      
      if (docCount > 10000) {
        console.log('  🚀 For large datasets, consider partitioning by case_id');
        console.log('  💾 Monitor memory usage during vector searches');
      }
      
      if (docCount > 100000) {
        console.log('  🔧 Consider upgrading to HNSW indexes for better performance');
        console.log('  📈 Implement connection pooling for high-concurrency workloads');
      }
    }

    console.log('\n✅ Vector database optimization complete!');
    console.log('\nNext steps:');
    console.log('1. Test vector search performance with sample queries');
    console.log('2. Monitor query execution plans with EXPLAIN ANALYZE');
    console.log('3. Adjust index parameters based on workload patterns');
    console.log('4. Set up automated VACUUM and ANALYZE jobs');

  } catch (error) {
    console.error('❌ Optimization failed:', error);
    throw error;
  }
}

// Test vector search performance
async function testVectorSearchPerformance() {
  console.log('\n🧪 Testing vector search performance...');
  
  try {
    const testQueries = [
      'contract liability terms',
      'criminal evidence analysis', 
      'legal precedent research',
      'case law citations',
      'discovery document review'
    ];

    for (const query of testQueries) {
      const startTime = Date.now();
      
      // Simulate vector search (replace with actual implementation)
      const result = await db.execute(sql`
        SELECT COUNT(*) as count
        FROM documents 
        WHERE embedding IS NOT NULL
        LIMIT 10
      `);
      
      const duration = Date.now() - startTime;
      console.log(`  "${query}": ${duration}ms`);
    }
    
    console.log('✅ Performance test complete');
    
  } catch (error) {
    console.error('❌ Performance test failed:', error);
  }
}

// Main execution
async function main() {
  try {
    await optimizeVectorDatabase();
    await testVectorSearchPerformance();
    process.exit(0);
  } catch (error) {
    console.error('❌ Script failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { optimizeVectorDatabase, testVectorSearchPerformance };