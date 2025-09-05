// scripts/embedding-worker.js
// Worker thread for parallel embedding generation

const { parentPort } = require('worker_threads');
const { pipeline } = require('@xenova/transformers');

let embedder = null;

// Initialize the embedding model
async function initializeEmbedder(model) {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', model);
  }
  return embedder;
}

// Generate embeddings for a batch of texts
async function generateEmbeddings(texts, model = 'Xenova/all-MiniLM-L6-v2') {
  const embedderInstance = await initializeEmbedder(model);
  const embeddings = [];
  
  for (const text of texts) {
    try {
      const output = await embedderInstance(text, { pooling: 'mean', normalize: true });
      embeddings.push(Array.from(output.data));
    } catch (error) {
      console.error('Embedding error:', error);
      embeddings.push(null);
    }
  }
  
  return embeddings;
}

// Handle messages from main thread
parentPort.on('message', async (message) => {
  try {
    if (message.type === 'generate') {
      const texts = message.errors.map(error => 
        `${error.type} ${error.file} ${error.message} ${error.context}`
      );
      
      const embeddings = await generateEmbeddings(texts, message.model);
      
      parentPort.postMessage({
        type: 'embeddings',
        embeddings: embeddings
      });
    }
  } catch (error) {
    parentPort.postMessage({
      type: 'error',
      error: error.message
    });
  }
});