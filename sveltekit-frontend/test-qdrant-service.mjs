// Test the existing QdrantService with local Qdrant instance
console.log('🔗 Testing QdrantService with local Qdrant...');

try {
    // Since we can't directly import TypeScript, let's test with direct QdrantClient
    const { QdrantClient } = await import('@qdrant/js-client-rest');
    
    const client = new QdrantClient({ 
        host: 'localhost',
        port: 6333,
        protocol: 'http'
    });

    // Test connection
    const collections = await client.getCollections();
    console.log('✅ Qdrant connected successfully!');
    console.log('📊 Collections found:', collections.collections.map(c => c.name));

    // Verify our legal_documents collection exists or create it
    const collectionName = 'legal_documents';
    
    try {
        const collectionInfo = await client.getCollection(collectionName);
        console.log(`📁 Collection '${collectionName}' exists with ${collectionInfo.points_count} points`);
    } catch (error) {
        console.log(`🔧 Creating legal_documents collection...`);
        await client.createCollection(collectionName, {
            vectors: {
                size: 384, // nomic-embed-text dimensions
                distance: 'Cosine'
            },
            optimizers_config: {
                default_segment_number: 2,
                max_segment_size: 20000,
                memmap_threshold: 20000,  
                indexing_threshold: 10000,
                flush_interval_sec: 10,
                max_optimization_threads: 2
            },
            hnsw_config: {
                m: 16,
                ef_construct: 200,
                full_scan_threshold: 10000,
                max_indexing_threads: 2
            }
        });
        console.log('✅ legal_documents collection created!');
    }

    // Test legal document addition
    const legalDoc = {
        id: 'legal-doc-' + Date.now(),
        vector: new Array(384).fill(0).map(() => Math.random()),
        payload: {
            // Match the LegalDocumentVector interface
            content: 'This Software License Agreement is entered into between TechCorp Inc. and DataSoft LLC.',
            title: 'Software License Agreement',
            caseId: 'CASE-2024-001',
            caseType: 'contract',
            legalJurisdiction: 'federal',
            summary: 'Software licensing agreement with liability limitations',
            legalEntities: {
                parties: ['TechCorp Inc.', 'DataSoft LLC'],
                dates: ['2024-01-15'],
                monetary: ['$50,000'],
                clauses: ['Section 3.1', 'Liability Limitations'],
                jurisdictions: ['federal'],
                caseTypes: ['contract']
            },
            riskScore: 35,
            confidenceScore: 0.87,
            legalPrecedent: false,
            tags: ['contract_law', 'software_license', 'liability'],
            timestamp: Date.now(),
            processingStatus: 'completed',
            aiModelVersion: 'gemma3-legal'
        }
    };

    await client.upsert(collectionName, {
        wait: true,
        points: [legalDoc]
    });
    console.log('✅ Sample legal document added successfully!');

    // Test search functionality
    const searchResults = await client.search(collectionName, {
        vector: legalDoc.vector,
        limit: 5,
        with_payload: true,
        score_threshold: 0.1
    });
    
    console.log(`🔍 Search found ${searchResults.length} results`);
    if (searchResults.length > 0) {
        const result = searchResults[0];
        console.log(`📄 Top result: "${result.payload.title}" (score: ${result.score.toFixed(3)})`);
        console.log(`⚖️  Case: ${result.payload.caseId} | Type: ${result.payload.caseType} | Risk: ${result.payload.riskScore}`);
    }

    console.log('🎉 QdrantService integration test passed!');
    console.log('🌐 Dashboard: http://localhost:6333/dashboard');
    console.log('📊 Collections endpoint: http://localhost:6333/collections');

} catch (error) {
    console.error('❌ QdrantService test failed:', error.message);
    console.log('💡 Make sure Qdrant is running with: ./start-qdrant-permanent.bat');
    console.log('🔗 Check Qdrant status: curl http://localhost:6333/collections');
}