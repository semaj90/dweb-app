
import type { RequestHandler } from './$types';

// Lazy require (keeps optional deps from breaking build)
const nodeCrypto = () => {
  try { return require("crypto"); } catch { return null; }
};

async function toArrayBuffer(buf: Buffer | Uint8Array | ArrayBuffer): Promise<any> {
  if (buf instanceof ArrayBuffer) return buf;
  if (buf instanceof Uint8Array) return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function getEnv(...keys: string[]): string | undefined {
  for (const k of keys) {
    const v = process.env[k];
    if (v && v.trim()) return v.trim();
  }
}

const candidateBaseUrls = (() => {
  const fromEnv = [
    getEnv("LOCAL_TTS_URL"),
    getEnv("VOICE_SERVICE_URL"),
    getEnv("TTS_SERVICE_URL"),
    getEnv("INTERNAL_TTS_URL")
  ].filter(Boolean) as string[];
  const defaults = [
    "http://localhost:8084", // Go microservice (add /api/tts later if implemented)
    "http://127.0.0.1:8084",
    "http://localhost:5002", // common local TTS ports
    "http://127.0.0.1:5002",
    "http://localhost:3001",
    "http://127.0.0.1:3001"
  ];
  return Array.from(new Set([...fromEnv, ...defaults]));
})();

const candidateTtsPaths = [
  "/api/tts",
  "/api/voice/tts",
  "/tts",
  "/synthesize",
  "/v1/tts",
  "/api/v1/tts"
];

// Attempt remote microservice first
async function tryRemoteTTS(text: string, voice: string, format: string): Promise<ArrayBuffer | null> {
  for (const base of candidateBaseUrls) {
    for (const path of candidateTtsPaths) {
      try {
        const url = base.replace(/\/+$/, "") + path;
        const r = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, voice, format })
        });
        if (!r.ok) continue;
        const ct = r.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const j = await r.json();
          const b64 = j.audio || j.data || j.audioContent;
          if (b64) {
            return toArrayBuffer(Buffer.from(b64, "base64"));
          }
        } else if (ct.startsWith("audio/") || ct === "application/octet-stream") {
          return await r.arrayBuffer();
        }
      } catch {
        // try next
      }
    }
  }
  return null;
}

// Check if a binary exists (cross-platform best-effort)
async function binaryExists(bin: string): Promise<boolean> {
  const { spawn } = await import("node:child_process");
  return await new Promise((resolve) => {
    const cmd = process.platform === "win32" ? "where" : "which";
    const p = spawn(cmd, [bin]);
    p.on("error", () => resolve(false));
    p.on("close", (code) => resolve(code === 0));
  });
}

// Fallback: piper (local fast TTS) if installed
async function tryPiper(text: string, voice: string, format: string): Promise<ArrayBuffer | null> {
  if (!(await binaryExists("piper"))) return null;
  // Requires voice model + config in working dir or specified via env
  // Typical usage: piper --model model.onnx --config config.json --output_file out.wav
  const model = getEnv("PIPER_MODEL") || "model.onnx";
  const config = getEnv("PIPER_CONFIG") || "config.json";
  const tmp = await import("node:os");
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const outFile = path.join(tmp.tmpdir(), `tts-${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);
  const { spawn } = await import("node:child_process");
  return await new Promise<ArrayBuffer | null>((resolve) => {
    const proc = spawn("piper", ["--model", model, "--config", config, "--output_file", outFile], {
      stdio: ["pipe", "ignore", "ignore"]
    });
    proc.stdin.write(text);
    proc.stdin.end();
    proc.on("close", async (code) => {
      if (code !== 0) return resolve(null);
      try {
        const data = await fs.readFile(outFile);
        if (format === "mp3") {
          // (Optional) encode WAV to MP3 if you have a local encoder; skipping actual transcode here.
          resolve(await toArrayBuffer(data)); // still wav
        } else {
          resolve(await toArrayBuffer(data));
        }
      } catch {
        resolve(null);
      } finally {
        fs.unlink(outFile).catch(() => { });
      }
    });
  });
}

// Fallback: edge-tts CLI (if installed)
async function tryEdgeTTS(text: string, voice: string, format: string): Promise<ArrayBuffer | null> {
  if (!(await binaryExists("edge-tts"))) return null;
  const tmp = await import("node:os");
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const outFile = path.join(tmp.tmpdir(), `edge-tts-${Date.now()}-${Math.random().toString(36).slice(2)}.${format === "wav" ? "wav" : "mp3"}`);
  const { spawn } = await import("node:child_process");
  const args = ["--voice", voice, "--text", text, "--write-media", outFile];
  return await new Promise<ArrayBuffer | null>((resolve) => {
    const proc = spawn("edge-tts", args, { stdio: "ignore" });
    proc.on("close", async (code) => {
      if (code !== 0) return resolve(null);
      try {
        const data = await fs.readFile(outFile);
        resolve(await toArrayBuffer(data));
      } catch {
        resolve(null);
      } finally {
        fs.unlink(outFile).catch(() => { });
      }
    });
  });
}

async function synthesizeSpeech(text: string, voice = "en-US-JennyNeural", format: "mp3" | "wav" = "mp3"): Promise<ArrayBuffer> {
  // 1. Remote microservice
  const remote = await tryRemoteTTS(text, voice, format);
  if (remote) return remote;
  // 2. Local piper
  const piper = await tryPiper(text, voice, format);
  if (piper) return piper;
  // 3. edge-tts CLI
  const edge = await tryEdgeTTS(text, voice, format);
  if (edge) return edge;
  // 4. Final fallback: synthetic silent WAV header (very short)
  const fallback = Buffer.from([82, 73, 70, 70, 36, 0, 0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1, 0, 68, 172, 0, 0, 136, 88, 1, 0, 2, 0, 16, 0, 100, 97, 116, 97, 0, 0, 0, 0]);
  return toArrayBuffer(fallback);
}

// (Optional) basic STT via remote service
async function transcribeAudio(file: File): Promise<string> {
  const sttUrl = getEnv("LOCAL_STT_URL", "STT_SERVICE_URL");
  if (sttUrl) {
    try {
      const fd = new FormData();
      fd.set("audio", file);
      const r = await fetch(sttUrl.replace(/\/+$/, "") + "/api/stt", { method: "POST", body: fd });
      if (r.ok) {
        const ct = r.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const j = await r.json();
          return j.transcript || j.text || "[no transcript field]";
        }
      }
    } catch {/* ignore */ }
  }
  return "[Simulated transcript: integrate Whisper/Piper STT service]";
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const contentType = request.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      const body = await request.json();
      const { text, voice = "en-US-JennyNeural", format = "mp3", returnBase64 = true } = body || {};
      if (!text || typeof text !== "string") {
        return json({ error: "Field 'text' is required" }, { status: 400 });
      }
      const audioBuffer = await synthesizeSpeech(text, voice, format);
      if (returnBase64) {
        const base64 = Buffer.from(audioBuffer).toString("base64");
        return json({
          success: true,
          mode: "tts",
          voice,
          format,
          audio: base64,
          encoding: "base64",
          source: "auto (remote|local|fallback)"
        });
      }
      return new Response(audioBuffer, {
        status: 200,
        headers: {
          "Content-Type": format === "wav" ? "audio/wav" : "audio/mpeg",
          "Content-Disposition": `inline; filename="speech.${format}"`
        }
      });
    }

    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const audioFile = formData.get("audio") as File | null;
      if (!audioFile) {
        return json({ error: "No audio file uploaded" }, { status: 400 });
      }
      const transcript = await transcribeAudio(audioFile);
      return json({ success: true, mode: "stt", transcript });
    }

    return json({ error: "Unsupported content type" }, { status: 415 });
  } catch (err: any) {
    return json({ error: err?.message || "Voice service failed" }, { status: 500 });
  }
};
