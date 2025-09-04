#!/usr/bin/env node
// Simple tokenizer.json generator fallback.
// Builds a trivial WordPiece-like vocab from repository source files if no tokenizer.json present.
import fs from 'fs';
import path from 'path';

const ROOT = process.env.REPO_ROOT || process.cwd();
const OUT = path.join(ROOT,'tokenizer.generated.json');

function gatherFiles(dir, collected=[]){
  const entries = fs.readdirSync(dir,{withFileTypes:true});
  for (const e of entries){
    if (e.name.startsWith('.') || e.name.includes('node_modules')) continue;
    const full = path.join(dir,e.name);
    if (e.isDirectory()) gatherFiles(full,collected);
    else if (/\.(js|ts|svelte|md|cjs|mjs|json)$/i.test(e.name)) collected.push(full);
  }
  return collected;
}

function buildVocab(files, limit=30000){
  const freq = new Map();
  for (const f of files){
    let text='';
    try { text = fs.readFileSync(f,'utf8'); } catch{ continue; }
    const tokens = text.split(/[^A-Za-z0-9_]+/).filter(Boolean);
    for (const t of tokens){
      const k = t.slice(0,64);
      freq.set(k,(freq.get(k)||0)+1);
    }
  }
  const sorted = [...freq.entries()].sort((a,b)=> b[1]-a[1]).slice(0,limit-5);
  const vocab = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]', ...sorted.map(x=>x[0])];
  const model = {
    version: 'legal-ai-fallback-1',
    truncation: { max_length: 512 },
    padding: { strategy: 'Longest', direction: 'Right', pad_to_multiple_of: null, pad_id:0, pad_token:'[PAD]' },
    added_tokens: [],
    normalizer: { type: 'Sequence', normalizers:[{type:'NFD'},{type:'StripAccents'}]},
    pre_tokenizer: { type:'Whitespace' },
    post_processor: { type:'TemplateProcessing', single:'$0', pair:'$A [SEP] $B:1', special_tokens:[['[SEP]',3]] },
    decoder: { type:'WordPiece', prefix:'##', cleanup:true },
    model: { type:'WordPiece', unk_token:'[UNK]', continuing_subword_prefix:'##', max_input_chars_per_word:100, vocab: Object.fromEntries(vocab.map((v,i)=>[v,i])) }
  };
  return model;
}

if (fs.existsSync(path.join(ROOT,'tokenizer.json'))){
  console.log('tokenizer.json exists - aborting (no overwrite)');
  process.exit(0);
}

console.log('Generating fallback tokenizer from source files...');
const files = gatherFiles(ROOT, []).slice(0,1000); // cap for speed
const model = buildVocab(files);
fs.writeFileSync(OUT, JSON.stringify(model,null,2));
console.log('Fallback tokenizer written to', OUT);
console.log('Set TOKENIZER_PATH=tokenizer.generated.json to use it, or rename to tokenizer.json.');
