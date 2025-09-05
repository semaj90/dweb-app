import fs from 'fs';
import FormData from 'form-data';
import fetch from 'node-fetch';

const target = process.env.OCR_URL || 'http://localhost:8601/api/ocr/extract';
const fileArg = process.argv[2];

// 1x1 transparent PNG binary (fallback)
const tinyPngBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=';
const SAMPLE_IMAGE_URL = process.env.SAMPLE_IMAGE_URL || 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/320px-Fronalpstock_big.jpg';

async function getTestFile() {
  if (fileArg && fs.existsSync(fileArg)) {
    return { stream: fs.createReadStream(fileArg), filename: fileArg.split(/[/\\]/).pop(), contentType: detectMime(fileArg) };
  }
  // Try to fetch a real sample image for better OCR signal
  try {
    const controller = new AbortController();
    const timeout = setTimeout(()=>controller.abort(), 8000);
    const resp = await fetch(SAMPLE_IMAGE_URL, { signal: controller.signal });
    clearTimeout(timeout);
    if (resp.ok) {
      const arrayBuf = await resp.arrayBuffer();
      const buf = Buffer.from(arrayBuf);
      if (buf.length > 1024) { // ensure not tiny
        return { buffer: buf, filename: 'sample.jpg', contentType: 'image/jpeg' };
      }
    }
  } catch (e) {
    // ignore and fallback
  }
  // Fallback: tiny PNG
  const buf = Buffer.from(tinyPngBase64, 'base64');
  return { buffer: buf, filename: 'tiny.png', contentType: 'image/png' };
}

function detectMime(name) {
  const ext = name.toLowerCase().split('.').pop();
  if (ext === 'png') return 'image/png';
  if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg';
  if (ext === 'gif') return 'image/gif';
  if (ext === 'pdf') return 'application/pdf';
  return 'application/octet-stream';
}

async function main() {
  const file = await getTestFile();
  console.log('Sending OCR test to', target, 'with file', file.filename, 'contentType', file.contentType);
  const fd = new FormData();
  if (file.stream) {
    fd.append('file', file.stream, { filename: file.filename, contentType: file.contentType });
  } else {
    fd.append('file', file.buffer, { filename: file.filename, contentType: file.contentType });
  }
  try {
    const res = await fetch(target, { method: 'POST', body: fd });
    const text = await res.text();
    console.log('Status:', res.status);
    console.log('Response:', text.slice(0, 1000));
  } catch (e) {
    console.error('OCR test failed:', e);
    process.exit(1);
  }
}
main();
