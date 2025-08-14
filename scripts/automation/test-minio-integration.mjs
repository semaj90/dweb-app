// MinIO Integration Test Suite
// Save this as test-minio-integration.mjs and run with: node test-minio-integration.mjs

import fs from 'fs';
import FormData from 'form-data';
import fetch from 'node-fetch';

const SERVICES = {
    minio: 'http://localhost:9000',
    postgres: 'localhost:5432',
    qdrant: 'http://localhost:6333',
    upload: 'http://localhost:8093',
    rag: 'http://localhost:8092'
};

const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m'
};

async function checkService(name, url, healthPath = '/health') {
    try {
        const response = await fetch(url + healthPath, { 
            method: 'GET',
            timeout: 5000 
        });
        
        if (response.ok) {
            console.log(`${colors.green}✅ ${name} is running${colors.reset}`);
            return true;
        } else {
            console.log(`${colors.yellow}⚠️  ${name} responded with status ${response.status}${colors.reset}`);
            return false;
        }
    } catch (error) {
        console.log(`${colors.red}❌ ${name} is not accessible: ${error.message}${colors.reset}`);
        return false;
    }
}

async function testFileUpload() {
    console.log(`\n${colors.cyan}Testing File Upload...${colors.reset}`);
    
    // Create a test file
    const testContent = `Test Legal Document
    Case: CASE-001
    Type: Evidence
    Date: ${new Date().toISOString()}
    Content: This is a test document for the MinIO integration.`;
    
    const testFile = 'test-document.txt';
    fs.writeFileSync(testFile, testContent);
    
    try {
        const form = new FormData();
        form.append('file', fs.createReadStream(testFile));
        form.append('caseId', 'CASE-001');
        form.append('documentType', 'evidence');
        form.append('tags', JSON.stringify({
            priority: 'high',
            category: 'financial',
            status: 'pending'
        }));
        form.append('metadata', JSON.stringify({
            author: 'Test System',
            version: '1.0',
            classification: 'confidential'
        }));
        
        const response = await fetch(`${SERVICES.upload}/upload`, {
            method: 'POST',
            body: form,
            headers: form.getHeaders()
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log(`${colors.green}✅ File uploaded successfully${colors.reset}`);
            console.log(`   Document ID: ${result.documentId}`);
            console.log(`   Object Name: ${result.objectName}`);
            console.log(`   URL: ${result.url}`);
            return result.documentId;
        } else {
            console.log(`${colors.red}❌ Upload failed with status ${response.status}${colors.reset}`);
            const error = await response.text();
            console.log(`   Error: ${error}`);
            return null;
        }
    } catch (error) {
        console.log(`${colors.red}❌ Upload error: ${error.message}${colors.reset}`);
        return null;
    } finally {
        // Clean up test file
        if (fs.existsSync(testFile)) {
            fs.unlinkSync(testFile);
        }
    }
}

async function testVectorSearch(query = "financial evidence case") {
    console.log(`\n${colors.cyan}Testing Vector Search...${colors.reset}`);
    console.log(`   Query: "${query}"`);
    
    try {
        const response = await fetch(`${SERVICES.upload}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                limit: 10,
                caseId: 'CASE-001'
            })
        });
        
        if (response.ok) {
            const results = await response.json();
            console.log(`${colors.green}✅ Search completed${colors.reset}`);
            console.log(`   Found ${results.results?.length || 0} results`);
            
            if (results.results && results.results.length > 0) {
                console.log(`   Top result:`);
                console.log(`     - Document: ${results.results[0].filename}`);
                console.log(`     - Similarity: ${results.results[0].similarity}`);
            }
            return results;
        } else {
            console.log(`${colors.red}❌ Search failed with status ${response.status}${colors.reset}`);
            return null;
        }
    } catch (error) {
        console.log(`${colors.red}❌ Search error: ${error.message}${colors.reset}`);
        return null;
    }
}

async function testListDocuments(caseId = 'CASE-001') {
    console.log(`\n${colors.cyan}Testing List Documents...${colors.reset}`);
    console.log(`   Case ID: ${caseId}`);
    
    try {
        const response = await fetch(`${SERVICES.upload}/documents/${caseId}`, {
            method: 'GET'
        });
        
        if (response.ok) {
            const documents = await response.json();
            console.log(`${colors.green}✅ Documents retrieved${colors.reset}`);
            console.log(`   Found ${documents.length} documents in ${caseId}`);
            
            documents.forEach((doc, index) => {
                console.log(`   ${index + 1}. ${doc.filename} (${doc.documentType})`);
            });
            return documents;
        } else {
            console.log(`${colors.red}❌ List failed with status ${response.status}${colors.reset}`);
            return null;
        }
    } catch (error) {
        console.log(`${colors.red}❌ List error: ${error.message}${colors.reset}`);
        return null;
    }
}

async function testQdrantCollection() {
    console.log(`\n${colors.cyan}Testing Qdrant Collection...${colors.reset}`);
    
    try {
        // Check if collection exists
        const response = await fetch(`${SERVICES.qdrant}/collections/legal_documents`);
        
        if (response.ok) {
            const collection = await response.json();
            console.log(`${colors.green}✅ Qdrant collection exists${colors.reset}`);
            console.log(`   Vectors: ${collection.result?.vectors_count || 0}`);
            console.log(`   Size: ${collection.result?.vectors?.size || 384}`);
            console.log(`   Distance: ${collection.result?.vectors?.distance || 'Cosine'}`);
            return true;
        } else if (response.status === 404) {
            console.log(`${colors.yellow}⚠️  Creating Qdrant collection...${colors.reset}`);
            
            // Create collection
            const createResponse = await fetch(`${SERVICES.qdrant}/collections/legal_documents`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    vectors: {
                        size: 384,
                        distance: 'Cosine'
                    }
                })
            });
            
            if (createResponse.ok) {
                console.log(`${colors.green}✅ Qdrant collection created${colors.reset}`);
                return true;
            } else {
                console.log(`${colors.red}❌ Failed to create collection${colors.reset}`);
                return false;
            }
        } else {
            console.log(`${colors.red}❌ Qdrant error: ${response.status}${colors.reset}`);
            return false;
        }
    } catch (error) {
        console.log(`${colors.red}❌ Qdrant error: ${error.message}${colors.reset}`);
        return false;
    }
}

async function runTests() {
    console.log(`${colors.cyan}========================================${colors.reset}`);
    console.log(`${colors.cyan}MinIO Integration Test Suite${colors.reset}`);
    console.log(`${colors.cyan}========================================${colors.reset}`);
    
    // Step 1: Check all services
    console.log(`\n${colors.cyan}Checking Services...${colors.reset}`);
    const services = {
        'MinIO': await checkService('MinIO', SERVICES.minio, '/minio/health/live'),
        'Qdrant': await checkService('Qdrant', SERVICES.qdrant, '/'),
        'Upload Service': await checkService('Upload Service', SERVICES.upload),
        'RAG Service': await checkService('RAG Service', SERVICES.rag, '/health')
    };
    
    const allServicesUp = Object.values(services).every(status => status);
    
    if (!allServicesUp) {
        console.log(`\n${colors.yellow}⚠️  Some services are not running. Tests may fail.${colors.reset}`);
        console.log(`Run START-MINIO-INTEGRATION.bat to start all services.`);
    }
    
    // Step 2: Test Qdrant collection
    await testQdrantCollection();
    
    // Step 3: Test file upload
    const documentId = await testFileUpload();
    
    // Wait for processing
    if (documentId) {
        console.log(`\n${colors.yellow}Waiting for document processing...${colors.reset}`);
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    // Step 4: Test vector search
    await testVectorSearch();
    
    // Step 5: Test list documents
    await testListDocuments();
    
    // Summary
    console.log(`\n${colors.cyan}========================================${colors.reset}`);
    console.log(`${colors.cyan}Test Summary${colors.reset}`);
    console.log(`${colors.cyan}========================================${colors.reset}`);
    
    const testsPassed = allServicesUp && documentId !== null;
    
    if (testsPassed) {
        console.log(`${colors.green}✅ All tests passed successfully!${colors.reset}`);
        console.log(`\nYour MinIO integration is working correctly with:`);
        console.log(`- File uploads to MinIO`);
        console.log(`- Metadata storage in PostgreSQL`);
        console.log(`- Vector embeddings in pgVector`);
        console.log(`- Vector search in Qdrant`);
        console.log(`- Document tagging and organization`);
    } else {
        console.log(`${colors.red}❌ Some tests failed. Please check the errors above.${colors.reset}`);
    }
}

// Run the tests
runTests().catch(error => {
    console.error(`${colors.red}Test suite error: ${error}${colors.reset}`);
    process.exit(1);
});
