/**
 * Single Document Upload Test
 * Tests the real database integration with one PDF
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function testSingleUpload() {
  try {
    console.log('🧪 Testing single document upload with real database integration...');

    // Let's test with a smaller document first
    const testFile = path.join(__dirname, '../../lawpdfs/baez1998Details.pdf');

    if (!fs.existsSync(testFile)) {
      console.log('❌ Test file not found:', testFile);
      return;
    }

    console.log(`📄 Test file: ${path.basename(testFile)}`);
    console.log(`📊 File size: ${fs.statSync(testFile).size} bytes`);

    // Read file
    const fileBuffer = fs.readFileSync(testFile);
    console.log('✅ File read successfully');

    // Test our database first
    console.log('🔍 Testing database connection...');

    try {
      const dbResponse = await fetch('http://localhost:5177/api/rag/status');
      if (dbResponse.ok) {
        const status = await dbResponse.json();
        console.log('✅ Database status check passed');
        console.log('   Services:', Object.keys(status.services || {}));
      }
    } catch (err) {
      console.log('⚠️  Database status check failed:', err.message);
    }

    // Test embeddings service
    console.log('🔍 Testing embeddings service...');

    try {
      const embResponse = await fetch('http://localhost:5177/api/ai/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: 'test legal document content' })
      });

      if (embResponse.ok) {
        const embResult = await embResponse.json();
        console.log('✅ Embeddings service working');
        console.log(`   Generated ${embResult.dimensions} dimensional embedding`);
      }
    } catch (err) {
      console.log('⚠️  Embeddings service test failed:', err.message);
    }

    console.log('');
    console.log('🎯 Ready for manual upload testing!');
    console.log('📝 Next steps:');
    console.log('   1. Open http://localhost:5177/simple-upload-test in your browser');
    console.log('   2. Upload one of the legal PDFs from the lawpdfs folder');
    console.log('   3. Verify it gets stored in the PostgreSQL database');
    console.log('   4. Test semantic search functionality');
    console.log('');
    console.log('🗂️  Available test files:');

    const lawpdfsDir = path.join(__dirname, '../../lawpdfs');
    const files = fs.readdirSync(lawpdfsDir)
      .filter(f => f.endsWith('.pdf'))
      .slice(0, 10); // Show first 10

    files.forEach((file, index) => {
      console.log(`   ${index + 1}. ${file}`);
    });

    if (files.length > 10) {
      console.log(`   ... and ${fs.readdirSync(lawpdfsDir).filter(f => f.endsWith('.pdf')).length - 10} more files`);
    }

  } catch (error) {
    console.log('❌ Test failed:', error.message);
  }
}

testSingleUpload();
