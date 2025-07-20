/**
 * Test script for nomic-embed-text model integration
 * Demonstrates the same functionality as the Python example
 */

async function testNomicEmbeddings() {
  console.log("🚀 Testing Nomic Embed Text model with Ollama...\n");

  // Test sentences (same as your Python example)
  const sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day",
  ];

  try {
    console.log("📝 Testing direct Ollama API call for embeddings:");
    sentences.forEach((sentence, i) => {
      console.log(`  ${i + 1}. "${sentence}"`);
    });
    console.log();

    // Generate embeddings for all sentences using direct Ollama API
    const embeddings = [];
    for (let i = 0; i < sentences.length; i++) {
      console.log(`⏳ Processing sentence ${i + 1}...`);
      const embedding = await getOllamaEmbedding(sentences[i]);
      embeddings.push(embedding);
      console.log(`✅ Generated embedding with ${embedding.length} dimensions`);
    }

    console.log(`\n🎯 Successfully generated ${embeddings.length} embeddings!`);
    console.log(`📏 Embedding dimensions: ${embeddings[0].length}`);

    // Calculate similarities between all pairs
    console.log("\n🔍 Calculating similarities between sentences:\n");

    const similarities = calculateSimilarityMatrix(embeddings);

    // Display similarity matrix
    console.log("Similarity Matrix:");
    console.log("================");

    // Header
    process.stdout.write("        ");
    for (let i = 0; i < sentences.length; i++) {
      process.stdout.write(`   S${i + 1}  `);
    }
    console.log();

    // Rows
    for (let i = 0; i < similarities.length; i++) {
      process.stdout.write(`   S${i + 1}   `);
      for (let j = 0; j < similarities[i].length; j++) {
        process.stdout.write(` ${similarities[i][j].toFixed(3)} `);
      }
      console.log();
    }

    console.log("\n📊 Similarity Analysis:");
    console.log("=====================");

    // Find most similar pairs
    const pairs = [];
    for (let i = 0; i < sentences.length; i++) {
      for (let j = i + 1; j < sentences.length; j++) {
        pairs.push({
          i,
          j,
          similarity: similarities[i][j],
          sentence1: sentences[i],
          sentence2: sentences[j],
        });
      }
    }

    // Sort by similarity (highest first)
    pairs.sort((a, b) => b.similarity - a.similarity);

    console.log("\nMost similar sentence pairs:");
    pairs.forEach((pair, index) => {
      const percentage = (pair.similarity * 100).toFixed(1);
      console.log(`${index + 1}. "${pair.sentence1}" ↔ "${pair.sentence2}"`);
      console.log(
        `   Similarity: ${pair.similarity.toFixed(4)} (${percentage}%)\n`
      );
    });

    console.log("🎉 Nomic Embed Text integration test completed successfully!");
  } catch (error) {
    console.error("❌ Error during embedding test:", error);
    throw error;
  }
}

/**
 * Get embedding from Ollama API directly
 */
async function getOllamaEmbedding(text) {
  const response = await fetch("http://localhost:11434/api/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "nomic-embed-text",
      prompt: text,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Ollama API error: ${response.status} ${response.statusText}`
    );
  }

  const data = await response.json();
  return data.embedding;
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Calculate similarity matrix for an array of embeddings
 */
function calculateSimilarityMatrix(embeddings) {
  const matrix = [];

  for (let i = 0; i < embeddings.length; i++) {
    const row = [];
    for (let j = 0; j < embeddings.length; j++) {
      if (i === j) {
        row.push(1.0); // Self-similarity is 1
      } else {
        row.push(cosineSimilarity(embeddings[i], embeddings[j]));
      }
    }
    matrix.push(row);
  }

  return matrix;
}

// Run the test
testNomicEmbeddings().catch(console.error);
