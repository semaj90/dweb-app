#!/usr/bin/env node

// Test script for GPU parse action via recommendation worker
const http = require('http');

const testPayload = {
  iteration: 1,
  timestamp: new Date().toISOString(),
  automatedRecommendations: [
    {
      action: 'run_gpu_parse',
      data: {
        content: JSON.stringify({
          contract: {
            title: "Sample Service Agreement",
            parties: ["Acme Corp", "Beta LLC"],
            terms: ["Payment within 30 days", "Service level 99.9%"]
          }
        }),
        format: 'json',
        options: {
          validation: true,
          performance_hints: true
        }
      }
    }
  ]
};

const postData = JSON.stringify(testPayload);

const options = {
  hostname: 'localhost',
  port: 4100,
  path: '/api/recommendations',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(postData)
  }
};

console.log('🧪 Testing GPU parse action via worker...');
console.log('📤 Payload:', JSON.stringify(testPayload, null, 2));

const req = http.request(options, (res) => {
  let data = '';

  res.on('data', (chunk) => {
    data += chunk;
  });

  res.on('end', () => {
    console.log('📥 Response status:', res.statusCode);
    console.log('📥 Response headers:', JSON.stringify(res.headers, null, 2));
    try {
      const response = JSON.parse(data);
      console.log('📥 Response body:', JSON.stringify(response, null, 2));
    } catch (e) {
      console.log('📥 Raw response:', data);
    }
  });
});

req.on('error', (e) => {
  console.error('❌ Request error:', e.message);
});

req.write(postData);
req.end();
