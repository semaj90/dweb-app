#!/usr/bin/env node
const { spawnSync } = require('child_process');
const chunkers=['chunk_go.cjs','chunk_ts.cjs','chunk_svelte.cjs','chunk_md.cjs'];
for(const c of chunkers){
  console.log('Running', c);
  const res=spawnSync('node',['scripts/chunkers/'+c],{stdio:'inherit'});
  if(res.status!==0){ console.error('Chunker failed', c); process.exit(res.status); }
}
console.log('All chunkers complete.');
