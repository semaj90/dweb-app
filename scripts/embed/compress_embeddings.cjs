#!/usr/bin/env node
/** Compress embeddings.jsonl to gzip and create sidecar checksums */
const fs=require('fs'); const path=require('path'); const zlib=require('zlib'); const crypto=require('crypto');
const EMB_DIR=path.resolve('.rag-metrics/embeddings');
const SRC=path.join(EMB_DIR,'embeddings.jsonl');
if(!fs.existsSync(SRC)) { console.error('Missing embeddings.jsonl'); process.exit(1);}
const gz=path.join(EMB_DIR,'embeddings.jsonl.gz');
const raw=fs.readFileSync(SRC);
fs.writeFileSync(gz, zlib.gzipSync(raw, {level:9}));
const hashes={
  raw_sha256: crypto.createHash('sha256').update(raw).digest('hex'),
  gz_sha256: crypto.createHash('sha256').update(fs.readFileSync(gz)).digest('hex')
};
fs.writeFileSync(path.join(EMB_DIR,'checksums.json'), JSON.stringify(hashes,null,2));
console.log('Compressed ->', gz, 'ratio', (raw.length / fs.statSync(gz).size).toFixed(2));
