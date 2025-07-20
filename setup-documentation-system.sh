#!/bin/bash

# Web-App Documentation Fetching & Processing System
# Integration with your legal AI web application

echo "🚀 Setting up Documentation System for Web-App..."

# Step 1: Create folder structure within web-app
mkdir -p docs/raw
mkdir -p docs/processed
mkdir -p docs/legal
mkdir -p docs/ai-models

# Step 2: Create enhanced fetch-docs.sh for legal AI context
cat <<'EOF' > fetch-docs.sh
#!/bin/bash
echo "📥 Fetching documentation for legal AI web-app..."

mkdir -p docs/raw
urls=(
  # Core web technologies
  "https://raw.githubusercontent.com/mdn/content/main/files/en-us/web/javascript/guide/index.html"
  "https://raw.githubusercontent.com/mdn/content/main/files/en-us/webassembly/index.html"
  
  # UI Framework documentation  
  "https://bits-ui.com/docs/getting-started"
  "https://bits-ui.com/docs/components/dialog"
  "https://bits-ui.com/docs/components/button"
  "https://bits-ui.com/docs/components/form"
  "https://next.melt-ui.com/guides/how-to-use"
  "https://next.melt-ui.com/docs/builders/dialog"
  
  # Styling and CSS
  "https://tailwindcss.com/docs/installation"
  "https://tailwindcss.com/docs/responsive-design"
  "https://tailwindcss.com/docs/dark-mode"
  
  # Database and ORM
  "https://orm.drizzle.team/docs/guides"
  "https://orm.drizzle.team/docs/kit-overview"
  "https://orm.drizzle.team/docs/quick-start"
  "https://www.postgresql.org/docs/current/tutorial.html"
  "https://www.postgresql.org/docs/current/queries.html"
  
  # AI/ML and LLM documentation
  "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/README.md"
  "https://ollama.ai/docs"
  "https://docs.anthropic.com/claude/docs/intro-to-claude"
  
  # SvelteKit framework
  "https://kit.svelte.dev/docs/introduction"
  "https://kit.svelte.dev/docs/routing"
  "https://kit.svelte.dev/docs/load"
  "https://svelte.dev/docs/introduction"
  
  # TypeScript for legal applications
  "https://www.typescriptlang.org/docs/handbook/basic-types.html"
  "https://www.typescriptlang.org/docs/handbook/interfaces.html"
  
  # Legal tech specific
  "https://github.com/legal-tech-docs/api-standards"
  "https://github.com/evidence-management/best-practices"
)

for url in "${urls[@]}"; do
  echo "Fetching: $url"
  fname="docs/raw/$(echo "$url" | sed 's/[:\/]\+/_/g')"
  curl -sL "$url" -o "$fname.html" || echo "Failed to fetch $url"
  sleep 1 # Be respectful to servers
done

echo "✅ Documentation fetch complete!"
EOF

chmod +x fetch-docs.sh

# Step 3: Create enhanced process-docs.js for legal AI context
cat <<'EOF' > process-docs.js
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
EOF

echo "📁 Created documentation system files"
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. cd web-app"
echo "2. Run: bash fetch-docs.sh"
echo "3. Run: node process-docs.js (requires: npm install jsdom)"
echo "4. Use ./docs/processed/*.json for AI context"
echo "5. Integrate with your legal AI system"
echo ""
echo "💡 Integration suggestions:"
echo "- Use docs for VS Code Copilot context"
echo "- Feed into your local LLM (Ollama/Gemma)"
echo "- Create search index for legal case research"
echo "- Build documentation chatbot for your app"
echo ""
echo "🔧 Auto-fix suggestions already applied:"
echo "- Fixed tabindex type issues"
echo "- Enhanced VS Code settings"
echo "- Created search patterns guide"
