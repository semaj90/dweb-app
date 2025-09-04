// Wrapper to set TEST_URL and run the main e2e upload test
const url = process.argv[2] || 'http://localhost:5173/api/production-upload';
process.env.TEST_URL = url;
import('./e2e-upload-test.mjs');
