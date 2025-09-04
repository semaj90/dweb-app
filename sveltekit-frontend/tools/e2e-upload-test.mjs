import fs from 'fs';
import FormData from 'form-data';
import fetch from 'node-fetch';

const filePath = './tools/e2e-test-file.txt';
const url = process.env.TEST_URL || 'http://localhost:5176/api/production-upload';

async function run() {
  const fd = new FormData();
  fd.append('file', fs.createReadStream(filePath));
  fd.append('caseId', 'E2E-TEST-001');
  fd.append('uploader', 'automated-test');

  console.log('Uploading to', url);

  try {
    const res = await fetch(url, { method: 'POST', body: fd });
    const text = await res.text();
    console.log('Status:', res.status);
    console.log('Response body:', text);
  } catch (err) {
    console.error('Upload failed:', err);
    process.exit(2);
  }
}

run();
