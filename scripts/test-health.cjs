const http = require('http');

const options = {
  hostname: 'localhost',
  port: 4100,
  path: '/health',
  method: 'GET'
};

console.log('Testing worker health...');

const req = http.request(options, (res) => {
  console.log(`Status: ${res.statusCode}`);
  let data = '';
  res.on('data', (chunk) => data += chunk);
  res.on('end', () => {
    console.log('Response:', data);
    process.exit(0);
  });
});

req.on('error', (e) => {
  console.error('Error:', e.message);
  process.exit(1);
});

req.setTimeout(5000, () => {
  console.log('Request timeout');
  req.destroy();
  process.exit(1);
});

req.end();
