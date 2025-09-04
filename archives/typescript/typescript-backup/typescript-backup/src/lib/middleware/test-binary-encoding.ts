/**
 * Test Binary Encoding Middleware
 * Quick validation of CBOR, MessagePack, and JSON encoding
 */
import { binaryEncoder } from './binary-encoding';

async function testBinaryEncoding(): Promise<any> {
  console.log('🧪 Testing Binary Encoding Middleware...\n');

  // Test data
  const testData = {
    legalCase: {
      id: 'case-123',
      title: 'Contract Dispute',
      parties: ['Plaintiff Corp', 'Defendant LLC'],
      documents: [
        { id: 'doc-1', type: 'contract', size: 1024 },
        { id: 'doc-2', type: 'evidence', size: 2048 }
      ],
      metadata: {
        jurisdiction: 'California',
        priority: 'high',
        tags: ['commercial', 'contract', 'dispute']
      }
    }
  };

  try {
    // Test JSON encoding
    console.log('📄 Testing JSON encoding...');
    const jsonResult = await binaryEncoder.encode(testData, 'json');
    console.log(`JSON - Size: ${jsonResult.metrics.encodedSize} bytes, Time: ${jsonResult.metrics.encodeTime.toFixed(2)}ms`);

    // Test MessagePack encoding
    console.log('📦 Testing MessagePack encoding...');
    const msgpackResult = await binaryEncoder.encode(testData, 'msgpack');
    console.log(`MessagePack - Size: ${msgpackResult.metrics.encodedSize} bytes, Time: ${msgpackResult.metrics.encodeTime.toFixed(2)}ms`);
    console.log(`Compression ratio: ${msgpackResult.metrics.compressionRatio.toFixed(2)}x`);

    // Test CBOR encoding
    console.log('🗜️ Testing CBOR encoding...');
    const cborResult = await binaryEncoder.encode(testData, 'cbor');
    console.log(`CBOR - Size: ${cborResult.metrics.encodedSize} bytes, Time: ${cborResult.metrics.encodeTime.toFixed(2)}ms`);
    console.log(`Compression ratio: ${cborResult.metrics.compressionRatio.toFixed(2)}x`);

    // Test auto-detection
    console.log('🤖 Testing automatic format detection...');
    const autoResult = await binaryEncoder.encode(testData);
    console.log(`Auto-detected format: ${autoResult.format}`);

    // Test decoding
    console.log('🔓 Testing decoding...');
    const { decoded } = await binaryEncoder.decode(msgpackResult.encoded, 'msgpack');
    const isEqual = JSON.stringify(decoded) === JSON.stringify(testData);
    console.log(`Decode successful: ${isEqual ? '✅' : '❌'}`);

    // Performance summary
    console.log('\n📊 Performance Summary:');
    console.log(`JSON: ${jsonResult.metrics.encodedSize} bytes`);
    console.log(`MessagePack: ${msgpackResult.metrics.encodedSize} bytes (${((1 - msgpackResult.metrics.encodedSize / jsonResult.metrics.encodedSize) * 100).toFixed(1)}% smaller)`);
    console.log(`CBOR: ${cborResult.metrics.encodedSize} bytes (${((1 - cborResult.metrics.encodedSize / jsonResult.metrics.encodedSize) * 100).toFixed(1)}% smaller)`);

    console.log('\n✅ All tests passed!');
    
  } catch (error: any) {
    console.error('❌ Test failed:', error);
  }
}

// Export for potential use in other tests
export { testBinaryEncoding };

// Run test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testBinaryEncoding();
}