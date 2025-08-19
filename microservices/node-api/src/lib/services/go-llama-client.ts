// Go LLM client adapter
// Calls a hypothetical Go microservice endpoint that processes combined GPU/WASM results.

export async function callGoLLM({ gpuResult, wasmResult }: { gpuResult: any, wasmResult: any }) {
  const endpoint = process.env.GO_LLAMA_URL || 'http://localhost:8096/process';
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gpuResult, wasmResult })
    });
    if (!res.ok) throw new Error(`Go LLM HTTP ${res.status}`);
    const data = await res.json();
    return data.result ?? data;
  } catch (e:any) {
    return { error: e.message };
  }
}
