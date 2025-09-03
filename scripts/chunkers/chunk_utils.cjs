#!/usr/bin/env node
/** Shared utilities for chunkers */
const crypto = require('crypto');
function sha256(str){ return crypto.createHash('sha256').update(str).digest('hex'); }
function estTokens(text){ return Math.ceil(text.split(/\s+/).length * 1.3); }
function splitWithOverlap(text, target=1000, overlap=120){
  const chunks=[]; let i=0; while(i<text.length){ const end=Math.min(text.length,i+target); const slice=text.slice(i,end); chunks.push(slice); if(end===text.length) break; i=end-overlap; if(i<0) i=0; }
  return chunks;
}
function stableChunkId(relPath, chunkIndex, content){
  return crypto.createHash('sha1').update(relPath+':'+chunkIndex+':'+sha256(content).slice(0,16)).digest('hex');
}
module.exports={sha256, estTokens, splitWithOverlap, stableChunkId};
