import { getEmbedding } from "../sveltekit-frontend/src/lib/server/services/embedding-service.js";

/**
 * Test script demonstrating the nomic-embed-text model integration
 * This mirrors the Python example you provided but uses our TypeScript implementation
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
    console.log("📝 Generating embeddings for sentences:");
    sentences.forEach((sentence, i) => {
      console.log(`  ${i + 1}. "${sentence}"`);
    });
    console.log();

    // Generate embeddings for all sentences
    const embeddings = [];
    for (let i = 0; i < sentences.length; i++) {
      console.log(`⏳ Processing sentence ${i + 1}...`);
      const embedding = await getEmbedding(sentences[i], "ollama");
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

    // Test with different types of content
    console.log("🧪 Testing with different content types:\n");

    const testCases = [
      {
        name: "Legal Text",
        text: "The defendant is charged with breach of contract under section 15 of the commercial law.",
      },
      {
        name: "Technical Text",
        text: "The database connection failed due to authentication timeout.",
      },
      {
        name: "Emotional Text",
        text: "She felt overwhelmed with joy and happiness.",
      },
      {
        name: "Factual Text",
        text: "The meeting is scheduled for Tuesday at 3 PM in conference room B.",
      },
    ];

    const testEmbeddings = [];
    for (const testCase of testCases) {
      console.log(`⏳ Processing ${testCase.name}...`);
      const embedding = await getEmbedding(testCase.text, "ollama");
      testEmbeddings.push({ ...testCase, embedding });
      console.log(`✅ Generated embedding for ${testCase.name}`);
    }

    // Compare test cases with original sentences
    console.log("\n🔗 Cross-content similarity analysis:");
    console.log("===================================");

    for (let i = 0; i < testEmbeddings.length; i++) {
      console.log(`\n${testEmbeddings[i].name}: "${testEmbeddings[i].text}"`);
      console.log("Similarities to original sentences:");

      for (let j = 0; j < sentences.length; j++) {
        const similarity = cosineSimilarity(
          testEmbeddings[i].embedding,
          embeddings[j]
        );
        const percentage = (similarity * 100).toFixed(1);
        console.log(
          `  → "${sentences[j]}": ${similarity.toFixed(4)} (${percentage}%)`
        );
      }
    }

    console.log(
      "\n🎉 Nomic Embed Text integration test completed successfully!"
    );
  } catch (error) {
    console.error("❌ Error during embedding test:", error);
    throw error;
  }
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
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
function calculateSimilarityMatrix(embeddings: number[][]): number[][] {
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
if (import.meta.url === `file://${process.argv[1]}`) {
  testNomicEmbeddings().catch(console.error);
}

export { calculateSimilarityMatrix, cosineSimilarity, testNomicEmbeddings };
