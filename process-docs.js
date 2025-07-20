import fs from 'fs';
import path from 'path';
import { JSDOM } from 'jsdom';

const rawDir = './docs/raw';
const outDir = './docs/processed';

if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

// Enhanced tag mapping for legal AI web-app
const tagMap = {
  javascript: ['vanilla-js', 'frontend', 'client-side', 'web-dev'],
  webassembly: ['wasm', 'performance', 'c++', 'native'],
  'bits-ui': ['svelte-ui', 'components', 'accessibility', 'forms'],
  'melt-ui': ['svelte-ui', 'headless', 'accessibility', 'dialogs'],
  tailwind: ['css', 'styling', 'responsive', 'dark-mode'],
  drizzle: ['orm', 'database', 'typescript', 'sql'],
  postgresql: ['database', 'sql', 'postgres', 'backend'],
  llama: ['llm', 'ai', 'local-ai', 'cpp'],
  ollama: ['llm', 'ai', 'model-serving', 'local'],
  anthropic: ['claude', 'ai', 'api', 'legal-ai'],
  svelte: ['framework', 'reactive', 'components'],
  sveltekit: ['framework', 'ssr', 'routing', 'fullstack'],
  typescript: ['types', 'safety', 'development'],
  legal: ['evidence', 'case-management', 'compliance']
};

// Process each document
fs.readdirSync(rawDir).forEach(file => {
  try {
    console.log(`Processing: ${file}`);
    
    const html = fs.readFileSync(path.join(rawDir, file), 'utf-8');
    const dom = new JSDOM(html);
    const doc = dom.window.document;
    
    // Extract content
    const title = doc.querySelector('title')?.textContent || file;
    const headings = Array.from(doc.querySelectorAll('h1, h2, h3')).map(h => h.textContent.trim());
    const paragraphs = Array.from(doc.querySelectorAll('p')).map(p => p.textContent.trim()).filter(p => p.length > 50);
    const codeBlocks = Array.from(doc.querySelectorAll('code, pre')).map(c => c.textContent.trim());
    
    // Clean and combine text
    const text = [
      title,
      ...headings.slice(0, 10),
      ...paragraphs.slice(0, 20),
      ...codeBlocks.slice(0, 5)
    ].join('\n').replace(/\s+/g, ' ').trim();
    
    // Determine tags
    let tags = [];
    for (const [key, keyTags] of Object.entries(tagMap)) {
      if (file.toLowerCase().includes(key)) {
        tags = tags.concat(keyTags);
      }
    }
    tags = Array.from(new Set(tags));
    
    // Add contextual tags based on content
    if (text.includes('component') || text.includes('props')) tags.push('components');
    if (text.includes('database') || text.includes('query')) tags.push('data');
    if (text.includes('authentication') || text.includes('security')) tags.push('security');
    if (text.includes('evidence') || text.includes('case') || text.includes('legal')) tags.push('legal-domain');
    
    // Create structured output
    const output = {
      id: file.replace('.html', ''),
      title,
      file,
      text: text.substring(0, 5000), // Limit length
      headings: headings.slice(0, 5),
      tags: Array.from(new Set(tags)),
      wordCount: text.split(' ').length,
      processedAt: new Date().toISOString(),
      relevance: tags.includes('legal-domain') ? 'high' : tags.includes('svelte-ui') ? 'medium' : 'low'
    };
    
    const outputPath = path.join(outDir, file.replace('.html', '.json'));
    fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
    
  } catch (error) {
    console.error(`Error processing ${file}:`, error.message);
  }
});

// Create index file
const processedFiles = fs.readdirSync(outDir)
  .filter(f => f.endsWith('.json'))
  .map(f => {
    const content = JSON.parse(fs.readFileSync(path.join(outDir, f), 'utf-8'));
    return {
      id: content.id,
      title: content.title,
      tags: content.tags,
      relevance: content.relevance,
      wordCount: content.wordCount
    };
  });

const index = {
  totalDocs: processedFiles.length,
  generatedAt: new Date().toISOString(),
  categories: {
    'legal-domain': processedFiles.filter(f => f.tags.includes('legal-domain')).length,
    'svelte-ui': processedFiles.filter(f => f.tags.includes('svelte-ui')).length,
    'database': processedFiles.filter(f => f.tags.includes('database')).length,
    'ai': processedFiles.filter(f => f.tags.includes('ai')).length
  },
  documents: processedFiles
};

fs.writeFileSync(path.join(outDir, 'index.json'), JSON.stringify(index, null, 2));

console.log(`✅ Processed ${processedFiles.length} documents`);
console.log('📊 Categories:', index.categories);
console.log('💾 Files saved to docs/processed/');
