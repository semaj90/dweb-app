import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { pool } from "../path/to/your/db/config"; // Update with your actual db config

async function main() {
  // Step 1: Configure the Tools
  const embeddings = new OpenAIEmbeddings({
    modelName: "nomic-embed-text",
    openAIApiKey: "N/A", // Not needed for local vLLM
    baseURL: "http://localhost:8000/v1", // Your local vLLM OpenAI-compatible endpoint
  });
  const vectorStore = new PGVectorStore({
    pool,
    tableName: "documents", // Your table name
    columns: {
      idColumnName: "id",
      vectorColumnName: "vector",
      contentColumnName: "content",
      metadataColumnName: "metadata",
    },
  });
  // Step 2: Load and Split a Document
  const loader = new TextLoader("./path/to/your/docs/some-doc.md");
  const rawDocs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splitDocs = await splitter.splitDocuments(rawDocs);
  // Step 3: Generate Embeddings and Store in DB
  await vectorStore.addDocuments(splitDocs);
  console.log("Documents have been indexed successfully!");
  await pool.end();
}

main().catch(console.error);
