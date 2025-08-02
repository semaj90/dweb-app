// Legal AI System - Integration Tests
// End-to-end tests for complete workflows

import { strict as assert } from 'assert';

// Test configuration
const config = {
    apiUrl: 'http://localhost:5173/api',
    timeout: 30000
};

// Test data
const testData = {
    case: {
        title: 'Integration Test Case - ' + Date.now(),
        description: 'Comprehensive test case for integration testing',
        status: 'open',
        priority: 'high',
        metadata: {
            test: true,
            timestamp: new Date().toISOString()
        }
    },
    document: {
        content: `This is a test legal document for integration testing.
                  It contains information about contract law, breach of contract,
                  and legal remedies. The document is used to test the vector
                  embedding and search functionality of the system.`,
        type: 'contract',
        metadata: {
            author: 'Test System',
            date: new Date().toISOString()
        }
    }
};

// Test results
const results = {
    passed: 0,
    failed: 0,
    workflows: []
};

// Helper functions
async function apiRequest(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        timeout: config.timeout
    };
    
    if (body) {
        options.body = JSON.stringify(body);
    }
    
    const response = await fetch(`${config.apiUrl}${endpoint}`, options);
    const data = await response.json();
    
    if (!response.ok) {
        throw new Error(`API Error: ${response.status} - ${data.error || 'Unknown error'}`);
    }
    
    return data;
}

function logWorkflow(name, status, steps, error = null) {
    const workflow = {
        name,
        status,
        steps,
        error: error ? error.message : null,
        timestamp: new Date().toISOString()
    };
    
    results.workflows.push(workflow);
    
    if (status === 'PASS') {
        results.passed++;
        console.log(`✓ ${name}`);
        if (steps.length > 0) {
            steps.forEach(step => console.log(`  - ${step}`));
        }
    } else {
        results.failed++;
        console.error(`✗ ${name}${error ? ': ' + error.message : ''}`);
    }
}

// Integration Test Workflows

// Workflow 1: Complete Case Management Flow
async function testCaseManagementWorkflow() {
    console.log('\n=== Case Management Workflow ===');
    
    const steps = [];
    let caseId;
    
    try {
        // Step 1: Create case
        const createdCase = await apiRequest('/cases', 'POST', testData.case);
        assert(createdCase.id, 'No case ID returned');
        caseId = createdCase.id;
        steps.push('Case created successfully');
        
        // Step 2: Add evidence
        const evidence = {
            case_id: caseId,
            type: 'document',
            description: 'Test evidence document',
            content: 'Important evidence content for the case'
        };
        
        const addedEvidence = await apiRequest('/evidence', 'POST', evidence);
        assert(addedEvidence.id, 'No evidence ID returned');
        steps.push('Evidence added to case');
        
        // Step 3: Score the case
        const scoreRequest = {
            case_id: caseId,
            case_data: createdCase,
            scoring_criteria: {
                evidence_strength: 0.8,
                witness_reliability: 0.7,
                legal_precedent: 0.9,
                public_interest: 0.6
            }
        };
        
        const scoreResult = await apiRequest('/case-scoring', 'POST', scoreRequest);
        assert(scoreResult.score >= 0 && scoreResult.score <= 100, 'Invalid score');
        steps.push(`Case scored: ${scoreResult.score}/100`);
        
        // Step 4: Generate AI summary
        const summaryRequest = {
            messages: [{
                role: 'user',
                content: `Summarize this case: ${createdCase.title} - ${createdCase.description}`
            }],
            context: {
                case_id: caseId,
                include_evidence: true
            }
        };
        
        const summary = await apiRequest('/ai/chat', 'POST', summaryRequest);
        assert(summary.response, 'No AI response received');
        steps.push('AI summary generated');
        
        // Step 5: Update case status
        const updateData = {
            status: 'closed',
            resolution: 'Test completed successfully',
            ai_summary: summary.response
        };
        
        const updated = await apiRequest(`/cases/${caseId}`, 'PATCH', updateData);
        assert(updated.status === 'closed', 'Case status not updated');
        steps.push('Case status updated to closed');
        
        // Step 6: Verify final state
        const finalCase = await apiRequest(`/cases/${caseId}`, 'GET');
        assert(finalCase.status === 'closed', 'Final case status incorrect');
        assert(finalCase.resolution, 'No resolution saved');
        steps.push('Final case state verified');
        
        logWorkflow('Case Management Workflow', 'PASS', steps);
    } catch (error) {
        logWorkflow('Case Management Workflow', 'FAIL', steps, error);
    }
}

// Workflow 2: Document Processing and Search
async function testDocumentWorkflow() {
    console.log('\n=== Document Processing Workflow ===');
    
    const steps = [];
    let documentId;
    
    try {
        // Step 1: Generate embedding
        const embeddingRequest = {
            content: testData.document.content,
            metadata: testData.document.metadata
        };
        
        const embedding = await apiRequest('/documents/embed', 'POST', embeddingRequest);
        assert(embedding.embedding, 'No embedding generated');
        assert(embedding.embedding.length === 384, 'Incorrect embedding dimensions');
        steps.push('Document embedding generated (384 dimensions)');
        
        // Step 2: Store document with vector
        const storeRequest = {
            document: {
                id: 'test-doc-' + Date.now(),
                ...testData.document
            },
            embedding: embedding.embedding
        };
        
        const stored = await apiRequest('/documents', 'POST', storeRequest);
        assert(stored.id, 'Document not stored');
        documentId = stored.id;
        steps.push('Document stored with vector');
        
        // Step 3: Search for similar documents
        const searchRequest = {
            query: 'contract breach legal remedies',
            limit: 10,
            threshold: 0.5
        };
        
        const searchResults = await apiRequest('/documents/search', 'POST', searchRequest);
        assert(searchResults.results, 'No search results');
        assert(searchResults.results.length > 0, 'No documents found');
        
        const found = searchResults.results.find(r => r.id === documentId);
        assert(found, 'Test document not found in search');
        assert(found.similarity > 0.7, 'Similarity score too low');
        steps.push(`Document found with similarity: ${found.similarity}`);
        
        // Step 4: Analyze document with AI
        const analysisRequest = {
            document_id: documentId,
            analysis_type: 'legal_review',
            options: {
                include_risks: true,
                include_recommendations: true
            }
        };
        
        const analysis = await apiRequest('/documents/analyze', 'POST', analysisRequest);
        assert(analysis.review, 'No analysis generated');
        steps.push('AI document analysis completed');
        
        // Step 5: Update document metadata
        const updateRequest = {
            metadata: {
                ...testData.document.metadata,
                analyzed: true,
                analysis_date: new Date().toISOString(),
                risk_level: 'low'
            }
        };
        
        const updated = await apiRequest(`/documents/${documentId}`, 'PATCH', updateRequest);
        assert(updated.metadata.analyzed === true, 'Metadata not updated');
        steps.push('Document metadata updated');
        
        logWorkflow('Document Processing Workflow', 'PASS', steps);
    } catch (error) {
        logWorkflow('Document Processing Workflow', 'FAIL', steps, error);
    }
}

// Workflow 3: Evidence Synthesis and Reporting
async function testEvidenceSynthesisWorkflow() {
    console.log('\n=== Evidence Synthesis Workflow ===');
    
    const steps = [];
    
    try {
        // Step 1: Create test case with multiple evidence items
        const testCase = await apiRequest('/cases', 'POST', {
            title: 'Evidence Synthesis Test Case',
            description: 'Testing evidence synthesis capabilities'
        });
        
        const caseId = testCase.id;
        steps.push('Test case created');
        
        // Step 2: Add multiple evidence items
        const evidenceItems = [
            {
                case_id: caseId,
                type: 'document',
                content: 'Financial records showing fraudulent transactions',
                relevance: 0.9
            },
            {
                case_id: caseId,
                type: 'testimony',
                content: 'Witness statement about suspicious behavior',
                relevance: 0.7
            },
            {
                case_id: caseId,
                type: 'physical',
                content: 'Computer hard drive with deleted files',
                relevance: 0.8
            }
        ];
        
        const addedEvidence = [];
        for (const evidence of evidenceItems) {
            const added = await apiRequest('/evidence', 'POST', evidence);
            addedEvidence.push(added);
        }
        steps.push(`Added ${addedEvidence.length} evidence items`);
        
        // Step 3: Synthesize evidence
        const synthesisRequest = {
            case_id: caseId,
            evidence_items: addedEvidence.map(e => ({
                id: e.id,
                type: e.type,
                content: e.content,
                relevance: e.relevance
            })),
            synthesis_type: 'comprehensive',
            options: {
                include_timeline: true,
                include_relationships: true,
                include_recommendations: true
            }
        };
        
        const synthesis = await apiRequest('/evidence/synthesize', 'POST', synthesisRequest);
        assert(synthesis.synthesis, 'No synthesis generated');
        assert(synthesis.timeline, 'No timeline generated');
        assert(synthesis.recommendations, 'No recommendations generated');
        steps.push('Evidence synthesis completed');
        
        // Step 4: Generate report
        const reportRequest = {
            case_id: caseId,
            report_type: 'case_summary',
            include_sections: [
                'executive_summary',
                'evidence_analysis',
                'timeline',
                'recommendations',
                'risk_assessment'
            ],
            format: 'markdown'
        };
        
        const report = await apiRequest('/reports/generate', 'POST', reportRequest);
        assert(report.content, 'No report content generated');
        assert(report.content.length > 500, 'Report too short');
        steps.push('Comprehensive report generated');
        
        // Step 5: Export report
        const exportRequest = {
            report_id: report.id,
            format: 'pdf',
            options: {
                include_metadata: true,
                include_appendices: true
            }
        };
        
        const exported = await apiRequest('/reports/export', 'POST', exportRequest);
        assert(exported.url || exported.data, 'No export data received');
        steps.push('Report exported successfully');
        
        logWorkflow('Evidence Synthesis Workflow', 'PASS', steps);
    } catch (error) {
        logWorkflow('Evidence Synthesis Workflow', 'FAIL', steps, error);
    }
}

// Workflow 4: Performance and Scalability Test
async function testPerformanceWorkflow() {
    console.log('\n=== Performance and Scalability Test ===');
    
    const steps = [];
    const metrics = {
        operations: 0,
        totalTime: 0,
        errors: 0
    };
    
    try {
        // Step 1: Batch document creation
        const batchSize = 20;
        const startTime = Date.now();
        
        const documents = Array(batchSize).fill(0).map((_, i) => ({
            content: `Test document ${i} with legal content about various topics including contracts, torts, and criminal law.`,
            metadata: { batch: true, index: i }
        }));
        
        const embeddings = await Promise.all(
            documents.map(doc => apiRequest('/documents/embed', 'POST', doc))
        );
        
        metrics.operations += batchSize;
        steps.push(`Generated ${batchSize} embeddings`);
        
        // Step 2: Parallel storage
        const stored = await Promise.all(
            embeddings.map((emb, i) => 
                apiRequest('/documents', 'POST', {
                    document: { id: `perf-test-${i}`, ...documents[i] },
                    embedding: emb.embedding
                })
            )
        );
        
        metrics.operations += batchSize;
        steps.push(`Stored ${stored.length} documents in parallel`);
        
        // Step 3: Concurrent searches
        const searchQueries = [
            'contract law',
            'criminal procedure',
            'tort liability',
            'legal remedies',
            'evidence rules'
        ];
        
        const searchResults = await Promise.all(
            searchQueries.map(query => 
                apiRequest('/documents/search', 'POST', { query, limit: 5 })
            )
        );
        
        metrics.operations += searchQueries.length;
        const totalResults = searchResults.reduce((sum, r) => sum + r.results.length, 0);
        steps.push(`Performed ${searchQueries.length} searches, found ${totalResults} results`);
        
        // Step 4: Stress test AI endpoint
        const aiRequests = Array(5).fill(0).map((_, i) => ({
            messages: [{
                role: 'user',
                content: `Quick legal question ${i}: What is a contract?`
            }],
            max_tokens: 100
        }));
        
        const aiResponses = await Promise.all(
            aiRequests.map(req => 
                apiRequest('/ai/chat', 'POST', req).catch(err => {
                    metrics.errors++;
                    return null;
                })
            )
        );
        
        const successfulAI = aiResponses.filter(r => r !== null).length;
        metrics.operations += successfulAI;
        steps.push(`AI requests: ${successfulAI}/${aiRequests.length} successful`);
        
        // Calculate metrics
        metrics.totalTime = Date.now() - startTime;
        const opsPerSecond = (metrics.operations / (metrics.totalTime / 1000)).toFixed(2);
        
        steps.push(`Total operations: ${metrics.operations}`);
        steps.push(`Total time: ${metrics.totalTime}ms`);
        steps.push(`Operations/second: ${opsPerSecond}`);
        steps.push(`Errors: ${metrics.errors}`);
        
        // Performance assertions
        assert(metrics.totalTime < 30000, 'Performance test took too long');
        assert(metrics.errors === 0, 'Errors occurred during performance test');
        assert(opsPerSecond > 1, 'Operations per second too low');
        
        logWorkflow('Performance and Scalability Test', 'PASS', steps);
    } catch (error) {
        logWorkflow('Performance and Scalability Test', 'FAIL', steps, error);
    }
}

// Main test runner
async function runIntegrationTests() {
    console.log('Legal AI System - Integration Tests');
    console.log('===================================\n');
    
    const startTime = Date.now();
    
    // Wait for services to be ready
    console.log('Waiting for services to be ready...');
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    try {
        // Check API health first
        const health = await apiRequest('/health', 'GET');
        console.log('API Health:', health);
        
        // Run workflows
        await testCaseManagementWorkflow();
        await testDocumentWorkflow();
        await testEvidenceSynthesisWorkflow();
        await testPerformanceWorkflow();
        
    } catch (error) {
        console.error('Integration test suite error:', error);
    }
    
    const duration = Date.now() - startTime;
    
    // Summary
    console.log('\n=== Integration Test Summary ===');
    console.log(`Total Workflows: ${results.passed + results.failed}`);
    console.log(`Passed: ${results.passed}`);
    console.log(`Failed: ${results.failed}`);
    console.log(`Duration: ${(duration / 1000).toFixed(2)}s`);
    
    // Save results
    const fs = await import('fs/promises');
    await fs.writeFile(
        'test-results-integration.json',
        JSON.stringify(results, null, 2)
    );
    
    // Exit with appropriate code
    process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runIntegrationTests();
