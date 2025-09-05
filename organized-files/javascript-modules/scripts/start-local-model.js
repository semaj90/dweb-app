import { spawn } from 'child_process';
const model = spawn('llama.cpp/main.exe', [
  '-m', 'models/gemma3-legal.gguf',
  '--server', '--port', '8080', '-ngl', '35'
]);
console.log('Model started on :8080');