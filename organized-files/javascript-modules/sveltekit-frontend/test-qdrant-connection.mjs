// Test Qdrant local connection
import { QdrantClient } from '@qdrant/js-client-rest';

async function testQdrantConnection() {
    console.log('🔗 Testing Qdrant local connection...');
    
    try {
        const client = new QdrantClient({ 
            host: 'localhost',
            port: 6333,
            protocol: 'http'
        });

        // Test basic connection
        const collections = await client.getCollections();
        console.log('✅ Qdrant connected successfully!');
        console.log('📊 Current collections:', collections.collections.length);

        // Test creating a collection for legal documents
        const collectionName = 'legal_documents_test';
        
        try {
            await client.getCollection(collectionName);
            console.log(`📁 Collection '${collectionName}' already exists`);
        } catch (error) {
            console.log(`🔧 Creating collection '${collectionName}'...`);
            await client.createCollection(collectionName, {
                vectors: {
                    size: 384, // nomic-embed-text dimensions
                    distance: 'Cosine'
                }
            });
            console.log('✅ Collection created successfully!');
        }

        // Test adding a sample legal document
        const sampleDoc = {
            id: 'test-doc-1',
            vector: new Array(384).fill(0).map(() => Math.random()),
            payload: {
                title: 'Sample Contract Agreement',
                content: 'This is a sample legal document for testing purposes.',
                caseType: 'contract',
                jurisdiction: 'federal',
                timestamp: Date.now()
            }
        };

        await client.upsert(collectionName, {
            wait: true,
            points: [sampleDoc]
        });
        console.log('✅ Sample document added successfully!');

        // Test searching
        const searchResults = await client.search(collectionName, {
            vector: sampleDoc.vector,
            limit: 5,
            with_payload: true
        });
        
        console.log('🔍 Search results:', searchResults.length);
        if (searchResults.length > 0) {
            console.log('📄 Found document:', searchResults[0].payload.title);
        }

        console.log('🎉 All Qdrant tests passed!');
        console.log('🌐 Dashboard available at: http://localhost:6333/dashboard');

    } catch (error) {
        console.error('❌ Qdrant connection failed:', error.message);
        console.log('💡 Make sure Qdrant is running: ./start-qdrant-permanent.bat');
    }
}

testQdrantConnection();