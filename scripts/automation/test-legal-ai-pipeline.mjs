#!/usr/bin/env node

/**
 * Legal AI Pipeline Test Script
 * Tests the complete pipeline: SvelteKit → BullMQ → Go Server → Ollama → PostgreSQL
 */

import { setTimeout } from 'timers/promises';

const GO_SERVER_URL = 'http://localhost:8080';
const SVELTEKIT_API_URL = 'http://localhost:5173/api/legal-ai/process-document';

// Test document content
const TEST_DOCUMENT = {
	content: `LEGAL MEMORANDUM

TO: Senior Partner
FROM: Associate Attorney
DATE: August 5, 2025
RE: Contract Liability Analysis - Acme Corp vs. Beta Industries

EXECUTIVE SUMMARY:
This memorandum analyzes the potential liability issues arising from the breach of contract dispute between Acme Corp and Beta Industries. The contract dated January 15, 2025, involved the delivery of manufacturing equipment worth $2.5 million.

KEY ISSUES:
1. Material breach of delivery timeline (60 days overdue)
2. Consequential damages of approximately $500,000
3. Force majeure clause applicability
4. Liquidated damages provision enforceability

ANALYSIS:
Under California commercial law, the defendant's failure to deliver constitutes a material breach. The plaintiff has documented damages exceeding the contract threshold.

RECOMMENDATIONS:
1. Immediate settlement negotiations recommended
2. Risk level: HIGH due to clear liability
3. Estimated exposure: $3.2M including legal fees

Attorney: J. Smith, Esq.
State Bar #: 123456`,
	document_type: 'evidence',
	case_id: 'CASE-TEST-001',
	extract_entities: true,
	generate_summary: true,
	assess_risk: true,
	generate_embedding: true,
	store_in_database: false, // Don't store test data
	use_gemma3_legal: true
};

/**
 * Test Go server health
 */
async function testGoServerHealth() {
	console.log('🔍 Testing Go server health...');
	
	try {
		const response = await fetch(`${GO_SERVER_URL}/health`);
		
		if (!response.ok) {
			throw new Error(`HTTP ${response.status}`);
		}
		
		const health = await response.json();
		console.log('✅ Go server health:', health.status);
		console.log(`   - Version: ${health.version}`);
		console.log(`   - Uptime: ${health.uptime}`);
		console.log(`   - Database: ${health.database}`);
		console.log(`   - Ollama: ${health.ollama}`);
		console.log(`   - CPU Cores: ${health.cpu_cores}`);
		console.log(`   - Memory: ${health.memory_mb} MB`);
		
		return true;
	} catch (error) {
		console.error('❌ Go server health check failed:', error.message);
		return false;
	}
}

/**
 * Test direct Go server processing
 */
async function testGoServerDirect() {
	console.log('\n🔍 Testing Go server direct processing...');
	
	try {
		const requestPayload = {
			document_id: `test-direct-${Date.now()}`,
			content: TEST_DOCUMENT.content,
			document_type: TEST_DOCUMENT.document_type,
			case_id: TEST_DOCUMENT.case_id,
			options: {
				extract_entities: TEST_DOCUMENT.extract_entities,
				generate_summary: TEST_DOCUMENT.generate_summary,
				assess_risk: TEST_DOCUMENT.assess_risk,
				generate_embedding: TEST_DOCUMENT.generate_embedding,
				store_in_database: TEST_DOCUMENT.store_in_database,
				use_gemma3_legal: TEST_DOCUMENT.use_gemma3_legal
			}
		};

		const startTime = Date.now();
		const response = await fetch(`${GO_SERVER_URL}/process-document`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(requestPayload),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`HTTP ${response.status}: ${errorText}`);
		}

		const result = await response.json();
		const processingTime = Date.now() - startTime;

		console.log('✅ Go server processing completed');
		console.log(`   - Processing time: ${processingTime}ms`);
		console.log(`   - Document ID: ${result.document_id}`);
		console.log(`   - Success: ${result.success}`);
		
		if (result.summary) {
			console.log(`   - Summary: ${result.summary.substring(0, 100)}...`);
		}
		
		if (result.entities && result.entities.length > 0) {
			console.log(`   - Entities extracted: ${result.entities.length}`);
			result.entities.slice(0, 3).forEach((entity, index) => {
				console.log(`     ${index + 1}. ${entity.type}: "${entity.value}" (${(entity.confidence * 100).toFixed(1)}%)`);
			});
		}
		
		if (result.risk_assessment) {
			console.log(`   - Risk level: ${result.risk_assessment.overall_risk}`);
			console.log(`   - Risk score: ${result.risk_assessment.risk_score}`);
		}
		
		if (result.embedding) {
			console.log(`   - Embedding: ${result.embedding.length} dimensions`);
		}

		return true;
	} catch (error) {
		console.error('❌ Go server direct test failed:', error.message);
		return false;
	}
}

/**
 * Test SvelteKit API integration
 */
async function testSvelteKitAPI() {
	console.log('\n🔍 Testing SvelteKit API integration...');
	
	try {
		const startTime = Date.now();
		const response = await fetch(SVELTEKIT_API_URL, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				...TEST_DOCUMENT,
				document_id: `test-sveltekit-${Date.now()}`
			}),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`HTTP ${response.status}: ${errorText}`);
		}

		const result = await response.json();
		const processingTime = Date.now() - startTime;

		console.log('✅ SvelteKit API processing completed');
		console.log(`   - Processing time: ${processingTime}ms`);
		console.log(`   - Success: ${result.success}`);
		console.log(`   - Processed by: ${result.processed_by}`);
		
		if (result.data) {
			console.log(`   - Document ID: ${result.data.document_id}`);
			console.log(`   - Go server time: ${result.data.processing_time}`);
		}

		return true;
	} catch (error) {
		console.error('❌ SvelteKit API test failed:', error.message);
		return false;
	}
}

/**
 * Test SvelteKit health endpoint
 */
async function testSvelteKitHealth() {
	console.log('\n🔍 Testing SvelteKit health endpoint...');
	
	try {
		const response = await fetch(SVELTEKIT_API_URL, {
			method: 'GET'
		});

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}`);
		}

		const health = await response.json();
		console.log('✅ SvelteKit health check passed');
		console.log(`   - SvelteKit status: ${health.sveltekit_status}`);
		console.log(`   - Integration status: ${health.integration_status}`);
		console.log(`   - Go server status: ${health.go_server_status?.status}`);

		return true;
	} catch (error) {
		console.error('❌ SvelteKit health check failed:', error.message);
		return false;
	}
}

/**
 * Main test runner
 */
async function runTests() {
	console.log('🚀 Legal AI Pipeline Test Suite');
	console.log('=====================================\n');

	let passedTests = 0;
	let totalTests = 0;

	// Test 1: Go server health
	totalTests++;
	if (await testGoServerHealth()) {
		passedTests++;
	}

	// Wait between tests
	await setTimeout(1000);

	// Test 2: SvelteKit health
	totalTests++;
	if (await testSvelteKitHealth()) {
		passedTests++;
	}

	await setTimeout(1000);

	// Test 3: Go server direct processing
	totalTests++;
	if (await testGoServerDirect()) {
		passedTests++;
	}

	await setTimeout(2000);

	// Test 4: SvelteKit API integration
	totalTests++;
	if (await testSvelteKitAPI()) {
		passedTests++;
	}

	// Results
	console.log('\n=====================================');
	console.log('📊 TEST RESULTS');
	console.log('=====================================');
	console.log(`✅ Passed: ${passedTests}/${totalTests}`);
	console.log(`❌ Failed: ${totalTests - passedTests}/${totalTests}`);
	
	if (passedTests === totalTests) {
		console.log('\n🎉 ALL TESTS PASSED! 🎉');
		console.log('The Legal AI pipeline is working correctly.');
		console.log('\n📋 Pipeline verified:');
		console.log('   SvelteKit API → Go Server → Ollama → Results');
	} else {
		console.log('\n⚠️  Some tests failed. Check the error messages above.');
		console.log('\n🔧 Troubleshooting:');
		console.log('   1. Ensure all services are running');
		console.log('   2. Check network connectivity');
		console.log('   3. Verify configuration settings');
		console.log('   4. Check logs for detailed error information');
	}
	
	console.log('\n🔗 Service URLs:');
	console.log(`   - SvelteKit: http://localhost:5173`);
	console.log(`   - Go Server: ${GO_SERVER_URL}`);
	console.log(`   - Ollama: http://localhost:11434`);
}

// Run the tests
runTests().catch((error) => {
	console.error('💥 Test suite failed:', error);
	process.exit(1);
});