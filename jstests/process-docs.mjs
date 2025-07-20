// Enhanced Documentation Processor for Legal AI Web-App
import fs from 'fs';
import path from 'path';

const rawDir = './docs/raw';
const outDir = './docs/processed';

// Ensure output directory exists
if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
}

// Enhanced tag mapping for legal AI context
const tagMap = {
    javascript: ['js', 'frontend', 'client-side', 'web-dev', 'dom'],
    webassembly: ['wasm', 'performance', 'native', 'cpp'],
    'bits-ui': ['svelte-ui', 'components', 'accessibility', 'forms', 'ui-lib'],
    'melt-ui': ['svelte-ui', 'headless', 'accessibility', 'state-management'],
    tailwind: ['css', 'styling', 'responsive', 'design-system'],
    drizzle: ['orm', 'database', 'typescript', 'sql', 'data-layer'],
    postgresql: ['database', 'sql', 'postgres', 'backend', 'data'],
    llama: ['llm', 'ai', 'local-ai', 'cpp', 'model-inference'],
    ollama: ['llm', 'ai', 'model-serving', 'local-deployment'],
    anthropic: ['claude', 'ai', 'api', 'legal-ai', 'nlp'],
    svelte: ['framework', 'reactive', 'components', 'spa'],
    sveltekit: ['framework', 'ssr', 'routing', 'fullstack'],
    typescript: ['types', 'safety', 'development', 'tooling'],
    legal: ['evidence', 'case-management', 'compliance', 'legal-tech']
};

function extractTextContent(htmlContent) {
    // Simple HTML text extraction (without DOM parser for basic functionality)
    return htmlContent
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
        .replace(/<[^>]*>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function determineRelevance(text, tags) {
    const legalKeywords = ['legal', 'evidence', 'case', 'court', 'law', 'attorney', 'document'];
    const uiKeywords = ['component', 'interface', 'user', 'form', 'dialog', 'accessibility'];
    const dataKeywords = ['database', 'query', 'data', 'sql', 'orm'];
    
    let score = 0;
    if (tags.some(tag => ['legal-domain', 'legal-tech'].includes(tag))) score += 10;
    if (tags.some(tag => ['svelte-ui', 'components'].includes(tag))) score += 8;
    if (tags.some(tag => ['database', 'data'].includes(tag))) score += 6;
    if (tags.some(tag => ['ai', 'llm'].includes(tag))) score += 7;
    
    const textLower = text.toLowerCase();
    legalKeywords.forEach(keyword => {
        if (textLower.includes(keyword)) score += 2;
    });
    
    if (score >= 10) return 'high';
    if (score >= 5) return 'medium';
    return 'low';
}

// Process documents
const processedDocs = [];

try {
    const files = fs.readdirSync(rawDir).filter(file => file.endsWith('.html'));
    
    if (files.length === 0) {
        console.log('❌ No HTML files found in docs/raw/');
        console.log('💡 Run fetch-docs.ps1 first to download documentation');
        process.exit(1);
    }
    
    console.log(`📚 Processing ${files.length} documentation files...`);
    
    files.forEach((file, index) => {
        console.log(`📄 Processing (${index + 1}/${files.length}): ${file}`);
        
        try {
            const filePath = path.join(rawDir, file);
            const htmlContent = fs.readFileSync(filePath, 'utf-8');
            const text = extractTextContent(htmlContent);
            
            // Determine tags based on filename and content
            let tags = [];
            Object.entries(tagMap).forEach(([key, keyTags]) => {
                if (file.toLowerCase().includes(key) || text.toLowerCase().includes(key)) {
                    tags = tags.concat(keyTags);
                }
            });
            
            // Add contextual tags
            const textLower = text.toLowerCase();
            if (textLower.includes('component') || textLower.includes('props')) tags.push('components');
            if (textLower.includes('database') || textLower.includes('query')) tags.push('data');
            if (textLower.includes('authentication') || textLower.includes('security')) tags.push('security');
            if (textLower.includes('evidence') || textLower.includes('legal')) tags.push('legal-domain');
            
            tags = [...new Set(tags)]; // Remove duplicates
            
            const doc = {
                id: file.replace('.html', ''),
                title: file.replace('.html', '').replace(/_/g, ' '),
                file,
                text: text.substring(0, 4000),
                tags,
                wordCount: text.split(' ').length,
                relevance: determineRelevance(text, tags),
                processedAt: new Date().toISOString(),
                source: 'web-fetch'
            };
            
            // Save individual document
            const outputPath = path.join(outDir, file.replace('.html', '.json'));
            fs.writeFileSync(outputPath, JSON.stringify(doc, null, 2));
            
            processedDocs.push(doc);
            
        } catch (error) {
            console.error(`❌ Error processing ${file}:`, error.message);
        }
    });
    
    // Create comprehensive index
    const index = {
        meta: {
            totalDocs: processedDocs.length,
            generatedAt: new Date().toISOString(),
            version: '1.0',
            purpose: 'Legal AI Web-App Documentation Index'
        },
        statistics: {
            byRelevance: {
                high: processedDocs.filter(d => d.relevance === 'high').length,
                medium: processedDocs.filter(d => d.relevance === 'medium').length,
                low: processedDocs.filter(d => d.relevance === 'low').length
            },
            byCategory: {
                'legal-domain': processedDocs.filter(d => d.tags.includes('legal-domain')).length,
                'svelte-ui': processedDocs.filter(d => d.tags.includes('svelte-ui')).length,
                'database': processedDocs.filter(d => d.tags.includes('database')).length,
                'ai': processedDocs.filter(d => d.tags.includes('ai')).length,
                'typescript': processedDocs.filter(d => d.tags.includes('typescript')).length
            },
            totalWords: processedDocs.reduce((sum, doc) => sum + doc.wordCount, 0)
        },
        documents: processedDocs.map(doc => ({
            id: doc.id,
            title: doc.title,
            tags: doc.tags,
            relevance: doc.relevance,
            wordCount: doc.wordCount
        }))
    };
    
    fs.writeFileSync(path.join(outDir, 'index.json'), JSON.stringify(index, null, 2));
    
    console.log(`\n✅ Successfully processed ${processedDocs.length} documents`);
    console.log('📊 Document Statistics:');
    console.log(`   📋 High relevance: ${index.statistics.byRelevance.high}`);
    console.log(`   📄 Medium relevance: ${index.statistics.byRelevance.medium}`);
    console.log(`   📃 Low relevance: ${index.statistics.byRelevance.low}`);
    console.log(`   📝 Total words: ${index.statistics.totalWords.toLocaleString()}`);
    console.log('\n📁 Files saved to docs/processed/');
    console.log('🎯 Ready for AI integration!');
    
} catch (error) {
    console.error('💥 Fatal error:', error.message);
    process.exit(1);
}
