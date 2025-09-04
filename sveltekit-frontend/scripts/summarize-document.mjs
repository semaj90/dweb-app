
import fs from 'fs';
import path from 'path';
import pdf from 'pdf-parse';
import { ChatOllama } from '@langchain/ollama';
import { HumanMessage } from '@langchain/core/messages';

const LAWPDFS_FOLDER = path.join(process.cwd(), 'lawpdfs');
const PDF_FILE = '100yearsPeople v. Jowy Omar Roman _ County of San Mateo, CA.pdf';

async function main() {
  console.log(`Attempting to summarize: ${PDF_FILE}`);

  const pdfPath = path.join(LAWPDFS_FOLDER, PDF_FILE);

  if (!fs.existsSync(pdfPath)) {
    console.error(`Error: File not found at ${pdfPath}`);
    console.error('Please ensure the file exists and the LAWPDFS_FOLDER is correct.');
    process.exit(1);
  }

  try {
    // 1. Read PDF and extract text
    console.log('Reading and parsing PDF file...');
    const dataBuffer = fs.readFileSync(pdfPath);
    const data = await pdf(dataBuffer);
    const text = data.text;
    console.log(`Successfully extracted ${text.length} characters of text.`);

    // 2. Summarize with LangChain.js and Ollama
    console.log('Connecting to Ollama and generating summary...');
    const ollama = new ChatOllama({
      baseUrl: "http://localhost:11434", // Default Ollama endpoint
      model: "gemma3-legal",
    });

    const prompt = new HumanMessage(`Please provide a concise summary of the following legal document:\n\n${text.substring(0, 4000)}`); // Summarize the first 4000 characters to be safe

    const response = await ollama.invoke([prompt]);

    console.log('\n--- Summary ---');
    console.log(response.content);
    console.log('--- End of Summary ---\n');

  } catch (error) {
    console.error('\n--- An error occurred ---');
    if (error.code === 'ECONNREFUSED') {
      console.error('Connection to Ollama failed. Is Ollama running at http://localhost:11434?');
    } else {
      console.error(error.message);
    }
    console.error('---------------------------\n');
    process.exit(1);
  }
}

main();
