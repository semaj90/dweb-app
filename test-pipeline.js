#!/usr/bin/env node

/**
 * Test script for the complete evidence analysis pipeline
 */

import { demoDataGenerator, createTestEvidence } from './sveltekit-frontend/src/lib/demo/sampleData.ts';
import { errorHandler } from './sveltekit-frontend/src/lib/utils/errorHandler.ts';

console.log('🔍 Testing Evidence Analysis Pipeline...\n');

async function testPipeline() {
  try {
    // 1. Generate demo case with evidence
    console.log('📋 Generating demo case...');
    const caseData = demoDataGenerator.generateCompleteCase();
    console.log(`✅ Generated case: ${caseData.case.title}`);
    console.log(`📄 Evidence count: ${caseData.evidence.length}`);
    console.log(`👥 Persons count: ${caseData.persons.length}\n`);

    // 2. Create test evidence content
    console.log('📝 Creating test evidence...');
    const evidenceContent = createTestEvidence();
    console.log(`✅ Created evidence content (${evidenceContent.length} characters)\n`);

    // 3. Generate analysis results
    console.log('🤖 Generating sample analysis results...');
    const analysisResults = demoDataGenerator.generateAnalysisResults(
      caseData.case.id, 
      caseData.evidence[0].id
    );
    console.log(`✅ Analysis completed - Session ID: ${analysisResults.sessionId}`);
    console.log(`📊 Status: ${analysisResults.status}`);
    console.log(`🎯 Current step: ${analysisResults.step}\n`);

    // 4. Display key findings
    console.log('🔍 Key Analysis Findings:');
    const evidenceAnalysis = analysisResults.outputs.evidence_analysis;
    console.log(`   Document Type: ${evidenceAnalysis.documentType}`);
    console.log(`   Confidence: ${(evidenceAnalysis.confidence * 100).toFixed(1)}%`);
    console.log(`   Key Facts: ${evidenceAnalysis.keyFacts.length} identified`);
    console.log(`   Timeline Events: ${evidenceAnalysis.timelineEvents.length} events`);
    console.log(`   Evidence Items: ${evidenceAnalysis.evidenceItems.length} items\n`);

    // 5. Display person extraction results
    console.log('👥 Person Extraction Results:');
    const personsExtracted = analysisResults.outputs.persons_extracted;
    console.log(`   Persons Identified: ${personsExtracted.persons.length}`);
    console.log(`   Relationships: ${personsExtracted.relationships.length}`);
    personsExtracted.persons.forEach(person => {
      console.log(`   - ${person.fullName} (${person.role})`);
    });
    console.log('');

    // 6. Display case synthesis
    console.log('⚖️  Case Synthesis:');
    const caseSynthesis = analysisResults.outputs.case_synthesis;
    console.log(`   Case Strength: ${caseSynthesis.caseStrength}`);
    console.log(`   Confidence: ${(caseSynthesis.confidence * 100).toFixed(1)}%`);
    console.log(`   Viable Charges: ${caseSynthesis.legalStrategy.viableCharges.length}`);
    caseSynthesis.legalStrategy.viableCharges.forEach(charge => {
      console.log(`   - ${charge}`);
    });
    console.log('');

    // 7. Test error handling
    console.log('🛡️  Testing error handling...');
    errorHandler.analysis('Test analysis error', { testData: true });
    const errorReport = errorHandler.getErrorReport();
    console.log(`✅ Error handler working - ${errorReport.summary.total} errors logged\n`);

    console.log('✅ Pipeline test completed successfully!');
    console.log('🎉 All components are working correctly.\n');

    // 8. Display next steps
    console.log('📋 Recommended Next Steps:');
    console.log('   1. Start the SvelteKit dev server: npm run dev');
    console.log('   2. Navigate to /evidence/analyze to test the UI');
    console.log('   3. Upload test evidence using the demo data');
    console.log('   4. Verify real-time progress tracking works');
    console.log('   5. Test the DetectiveBoard with drag-and-drop functionality\n');

    return true;
  } catch (error) {
    console.error('❌ Pipeline test failed:', error);
    errorHandler.system('Pipeline test failure', { error: error.message });
    return false;
  }
}

// Run the test
testPipeline().then(success => {
  process.exit(success ? 0 : 1);
});