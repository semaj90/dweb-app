// OCR Language Extract API
// Provides OCR and text extraction capabilities with language detection

import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
// Correct tesseract import (library present as tesseract.js in package.json)
import { createWorker } from 'tesseract.js';
// Lazy dynamic imports for optional dependencies (avoid crash if missing during partial installs)
let francFn: ((text: string) => string) | null = null;
let prom: any = null;

// Metrics (prom-client) initialization (lazy)
async function initMetrics() {
  if (prom) return prom;
  try {
    prom = await import('prom-client');
    prom.register.setDefaultLabels({ service: 'ocr-langextract' });
    metrics.ocrRequests = new prom.Counter({ name: 'ocr_request_total', help: 'Total OCR requests', labelNames: ['result'] });
    metrics.ocrLatency = new prom.Histogram({ name: 'ocr_latency_seconds', help: 'OCR end-to-end latency', buckets: [0.25, 0.5, 0.75, 1, 2, 3, 5] });
    metrics.preprocessFailures = new prom.Counter({ name: 'ocr_preprocess_fail_total', help: 'Preprocessing failures' });
  } catch (_e) {
    // Silently ignore; metrics optional
  }
  return prom;
}

const metrics: { [k: string]: any } = {};

// Language detection dynamic loader
async function detectLanguageFranc(text: string): Promise<string | null> {
  if (!text || text.trim().length < 20) return null;
  if (!francFn) {
    try {
      const mod: any = await import('franc');
      francFn = mod.franc || mod.default || null;
    } catch (_e) {
      francFn = null;
    }
  }
  try {
    if (francFn) {
      const code = francFn(text);
      if (code && code !== 'und') return code; // ISO 639-3 code
    }
  } catch (_e) {/* ignore */ }
  return null;
}

// Worker pool implementation -------------------------------------------------
interface PooledWorker { id: number; busy: boolean; worker: any; }
const MAX_WORKERS = parseInt(process.env.OCR_MAX_WORKERS || '1', 10);
const workerPool: PooledWorker[] = [];
const waitQueue: { resolve: (w: PooledWorker) => void; reject: (e: any) => void; timeout: NodeJS.Timeout }[] = [];

async function initWorker(langs: string): Promise<any> {
  return createWorker(langs);
}

async function acquireWorker(langs: string, timeoutMs = 15000): Promise<PooledWorker> {
  // Try idle
  for (const w of workerPool) {
    if (!w.busy) { w.busy = true; return w; }
  }
  // Create new if capacity
  if (workerPool.length < MAX_WORKERS) {
    const worker = await initWorker(langs);
    const pooled: PooledWorker = { id: workerPool.length, busy: true, worker };
    workerPool.push(pooled);
    return pooled;
  }
  // Queue
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      const idx = waitQueue.findIndex(q => q.resolve === resolve);
      if (idx >= 0) waitQueue.splice(idx, 1);
      reject(new Error('WORKER_ACQUIRE_TIMEOUT'));
    }, timeoutMs);
    waitQueue.push({ resolve, reject, timeout });
  });
}

function releaseWorker(pw: PooledWorker) {
  pw.busy = false;
  // Serve queue
  while (waitQueue.length) {
    const q = waitQueue.shift();
    if (!q) break;
    clearTimeout(q.timeout);
    if (pw.busy) continue; // just in case
    pw.busy = true;
    q.resolve(pw);
    return;
  }
}

// Security helpers -----------------------------------------------------------
function requireApiKey(request: Request): boolean {
  const expected = process.env.OCR_API_KEY;
  if (!expected) return true; // feature disabled
  const provided = request.headers.get('x-api-key') || request.headers.get('authorization');
  return !!provided && provided.replace(/^[Bb]earer\s+/, '') === expected;
}

const MAX_FILE_BYTES = parseInt(process.env.OCR_MAX_FILE_BYTES || `${10 * 1024 * 1024}`, 10); // 10MB default

// Utility: basic magic byte check for common formats
function looksLikeImage(buf: Buffer): boolean {
  if (buf.length < 4) return false;
  const sig = buf.slice(0, 4).toString('hex');
  return sig.startsWith('ffd8') || sig === '89504e47' || sig === '47494638' || sig.startsWith('424d'); // JPEG/PNG/GIF/BMP
}

  try {
    const formData = await request.formData();
    await initMetrics();

    if (!requireApiKey(request)) {
      return json({ error: 'Unauthorized', code: 'UNAUTHORIZED' }, { status: 401 });
    }

    const imageFile = formData.get("image") as File;
    const languages = (formData.get("languages") as string) || "eng";
    const preprocessImage = formData.get("preprocess") === "true";
    const preprocessParam = (formData.get("preprocess") as string) || ""; // e.g. grayscale,normalize,sharpen
    const preprocessModes = preprocessParam.split(',').map(s => s.trim()).filter(Boolean);
    if (!imageFile) {
      return json({ error: "Image file is required" }, { status: 400 });
      return json({ error: "Image file is required", code: 'NO_FILE' }, { status: 400 });

    // Validate file type
    if (!imageFile.type.startsWith("image/")) {
      return json({ error: "File must be an image" }, { status: 400 });
      return json({ error: "File must be an image", code: 'INVALID_TYPE' }, { status: 400 });


      if (imageFile.size > MAX_FILE_BYTES) {
        return json({ error: `File exceeds max size ${MAX_FILE_BYTES} bytes`, code: 'FILE_TOO_LARGE' }, { status: 413 });
    }
    const startTime = Date.now();

    // Convert file to buffer
    const arrayBuffer = await imageFile.arrayBuffer();
    let buffer = Buffer.from(arrayBuffer); // Fixed: Use Buffer.from() to ensure proper type


      if (!looksLikeImage(buffer)) {
        return json({ error: 'Magic bytes do not resemble a supported image', code: 'BAD_MAGIC' }, { status: 400 });
      }
    // Preprocess image if requested
    if (preprocessImage) {
      let imageMeta: { width?: number; height?: number } = {};
      if (preprocessModes.length) {
        try {
          const prep = await preprocessImageBuffer(buffer, preprocessModes);
          buffer = prep.buffer;
          imageMeta = prep.meta;
        } catch (e) {
          metrics.preprocessFailures && metrics.preprocessFailures.inc();
        }

    // Parse languages parameter
    const langs = languages.split(",").map((lang) => lang.trim());

      const joinedLangs = langs.join('+') || 'eng';
    // Perform OCR with fixed Tesseract configuration
      // Perform OCR (worker pool)
      const ocrResult = await performOCR(buffer, joinedLangs, imageMeta);
    const processingTime = Date.now() - startTime;


      metrics.ocrRequests && metrics.ocrRequests.inc({ result: 'success' });
      metrics.ocrLatency && metrics.ocrLatency.observe(processingTime / 1000);
    return json({
      success: true,
      result: {
        text: ocrResult.text,
        confidence: ocrResult.confidence,
        languages: langs,
        languages: langs,
        detectedLanguage: ocrResult.detectedLanguage,
        characterCount: ocrResult.text.length,
        blocks: ocrResult.blocks || [],
        paragraphs: ocrResult.paragraphs || [],
        lines: ocrResult.lines || [],
        words: ocrResult.words || [],
        words: ocrResult.words || [],
        normalizedWordBoxes: ocrResult.normalizedWordBoxes || []
      metadata: {
        originalFileName: imageFile.name,
        fileSize: imageFile.size,
        mimeType: imageFile.type,
        preprocessed: preprocessImage,
        preprocessed: preprocessModes.length > 0 ? preprocessModes : false,
        tesseractVersion: "5.0.0",
        tesseractVersion: ocrResult.version || undefined,
    });
  } catch (error: any) {
    console.error("OCR processing error:", error);
    return json(
      metrics.ocrRequests && metrics.ocrRequests.inc({ result: 'error' });
      { error: "OCR processing failed", details: error.message },
    { error: "OCR processing failed", code: error.code || 'OCR_FAIL', details: error.message },
    );
  }
};

      export const GET: RequestHandler = async ({ url, request }): Promise<any> => {
  try {
    const action = url.searchParams.get("action");

    switch (action) {
      case "supported_languages":
        return json({
          languages: getSupportedLanguages(),
          total: getSupportedLanguages().length,
        });

      case "health": {
        const healthCheck = await performHealthCheck();
        return json(healthCheck);
      }
      case "metrics": {
        await initMetrics();
        if (!prom) return json({ error: 'Metrics not available' }, { status: 503 });
        return new Response(prom.register.metrics(), { status: 200, headers: { 'Content-Type': prom.register.contentType } });
      }

      case "capabilities":
        return json({
          features: [
            "Text extraction",
            "Multi-language support",
            "Image preprocessing",
            "Confidence scoring",
            "Layout analysis",
            "Word/line/paragraph detection",
            "Worker pool",
            "Normalized bounding boxes",
          ],
          supportedFormats: ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
          maxFileSize: "10MB",
          languages: getSupportedLanguages().length,
          workerPool: { max: MAX_WORKERS, current: workerPool.length, busy: workerPool.filter(w => w.busy).length },
        });

      default:
        return json({
          message: "OCR Language Extract API",
          version: "2.0.0",
          availableActions: ["supported_languages", "health", "capabilities", "metrics"],
        });
    }
  } catch (error: any) {
    console.error("OCR API error:", error);
    return json(
      { error: "OCR API request failed", details: error.message },
      { status: 500 }
    );
  }
};

// Core OCR functions

async function performOCR(
  buffer: Buffer,
  languages: string, // joined with +
  imageMeta: { width?: number; height?: number }
): Promise<{
  text: string;
  confidence: number;
  detectedLanguage: string;
  blocks?: any[];
  paragraphs?: any[];
  lines?: any[];
  words?: any[];
  normalizedWordBoxes?: Array<{ x: number; y: number; w: number; h: number; text: string; confidence: number }>;
  version?: string;
}> {
  try {
    const pooled = await acquireWorker(languages);
    let data: any = {};
    try {
      const result = await pooled.worker.recognize(buffer);
      data = result.data || {};
    } finally {
      releaseWorker(pooled);
    }

    return {
      text: data.text || "",
      confidence: data.confidence || 0,
      detectedLanguage: await chooseLanguage(data.text || ""),
      blocks: data.blocks || [],
      paragraphs: (data as any).paragraphs || [],
      lines: (data as any).lines || [],
      words: (data as any).words || [],
      normalizedWordBoxes: normalizeBoxes((data as any).words || [], imageMeta),
    };
  } catch (error: any) {
    console.error("Tesseract OCR error:", error);

    // Fallback for OCR failure
    return {
      text: "",
      confidence: 0,
      detectedLanguage: "unknown",
      blocks: [],
      paragraphs: [],
      lines: [],
      words: [],
    };
  }
}

// Using broad Buffer type; casting sharp output to Buffer to satisfy TS
      async function preprocessImageBuffer(inputBuffer: Buffer, modes: string[]): Promise<{ buffer: Buffer; meta: { width?: number; height?: number } }> {
  try {
    // Use dynamic import for sharp to handle optional dependency
    const sharpMod = await import("sharp");
    const sharp = (sharpMod as any).default || sharpMod; // support both ESM/CJS
    let pipeline = sharp(inputBuffer);
    const meta = await pipeline.metadata();
    if (modes.includes('grayscale')) pipeline = pipeline.grayscale();
    if (modes.includes('normalize')) pipeline = pipeline.normalize();
    if (modes.includes('sharpen')) pipeline = pipeline.sharpen();
    if (modes.includes('threshold')) pipeline = pipeline.threshold();
    // Avoid lossy re-encode if original is png and no jpeg request
    if (modes.includes('jpeg') && meta.format !== 'jpeg') {
      pipeline = pipeline.jpeg({ quality: 90 });
    }
    const processedBuffer = await pipeline.toBuffer();
    return { buffer: processedBuffer as Buffer, meta: { width: meta.width, height: meta.height } };
  } catch (error: any) {
    console.warn("Image preprocessing failed, using original:", error);
    return { buffer: inputBuffer, meta: {} };
  }
      }

      async function chooseLanguage(text: string): Promise<string> {
        const francCode = await detectLanguageFranc(text);
        if (francCode) return francCode; // ISO 639-3
        return detectPrimaryLanguage(text); // fallback heuristic
}

function detectPrimaryLanguage(text: string): string {
  if (!text || text.trim().length === 0) {
    return "unknown";
  }
  const textSample = text.substring(0, 500).toLowerCase();
  const patterns: Record<string, RegExp> = {
    spanish: /[ñáéíóúü]/g,
    french: /[àâäçéèêëïîôùûüÿ]/g,
    german: /[äöüß]/g,
    italian: /[àèéìíîòóù]/g,
    portuguese: /[ãõçáàâêéíóôú]/g,
    russian: /[а-я]/g,
    chinese: /[\u4e00-\u9fff]/g,
    japanese: /[\u3040-\u309f\u30a0-\u30ff]/g,
    korean: /[\uac00-\ud7af]/g,
    arabic: /[\u0600-\u06ff]/g,
  };
  let maxMatches = 0;
  let detectedLang = "english";
  for (const [lang, pattern] of Object.entries(patterns)) {
    const matches = (textSample.match(pattern) || []).length;
    if (matches > maxMatches) {
      maxMatches = matches;
      detectedLang = lang;
    }
  }
  return detectedLang;
}

function getSupportedLanguages(): Array<{
  code: string;
  name: string;
  native: string;
}> {
  return [
    { code: "afr", name: "Afrikaans", native: "Afrikaans" },
    { code: "amh", name: "Amharic", native: "አማርኛ" },
    { code: "ara", name: "Arabic", native: "العربية" },
    { code: "asm", name: "Assamese", native: "অসমীয়া" },
    { code: "aze", name: "Azerbaijani", native: "azərbaycan dili" },
    { code: "bel", name: "Belarusian", native: "беларуская мова" },
    { code: "ben", name: "Bengali", native: "বাংলা" },
    { code: "bod", name: "Tibetan", native: "བོད་ཡིག" },
    { code: "bos", name: "Bosnian", native: "bosanski jezik" },
    { code: "bul", name: "Bulgarian", native: "български език" },
    { code: "cat", name: "Catalan", native: "català" },
    { code: "ceb", name: "Cebuano", native: "Cebuano" },
    { code: "ces", name: "Czech", native: "čeština" },
    { code: "chi_sim", name: "Chinese Simplified", native: "中文（简体）" },
    { code: "chi_tra", name: "Chinese Traditional", native: "中文（繁體）" },
    { code: "chr", name: "Cherokee", native: "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ" },
    { code: "cym", name: "Welsh", native: "Cymraeg" },
    { code: "dan", name: "Danish", native: "dansk" },
    { code: "deu", name: "German", native: "Deutsch" },
    { code: "dzo", name: "Dzongkha", native: "རྫོང་ཁ" },
    { code: "ell", name: "Greek", native: "Ελληνικά" },
    { code: "eng", name: "English", native: "English" },
    { code: "enm", name: "English Middle", native: "English (Middle)" },
    { code: "epo", name: "Esperanto", native: "Esperanto" },
    { code: "est", name: "Estonian", native: "eesti keel" },
    { code: "eus", name: "Basque", native: "euskera" },
    { code: "fas", name: "Persian", native: "فارسی" },
    { code: "fin", name: "Finnish", native: "suomi" },
    { code: "fra", name: "French", native: "français" },
    { code: "frk", name: "German Fraktur", native: "Deutsch (Fraktur)" },
    { code: "frm", name: "French Middle", native: "français (Middle)" },
    { code: "gle", name: "Irish", native: "Gaeilge" },
    { code: "glg", name: "Galician", native: "galego" },
    { code: "grc", name: "Greek Ancient", native: "Ἀρχαία ἑλληνικὴ" },
    { code: "guj", name: "Gujarati", native: "ગુજરાતી" },
    { code: "hat", name: "Haitian Creole", native: "Kreyòl ayisyen" },
    { code: "heb", name: "Hebrew", native: "עברית" },
    { code: "hin", name: "Hindi", native: "हिन्दी" },
    { code: "hrv", name: "Croatian", native: "hrvatski jezik" },
    { code: "hun", name: "Hungarian", native: "magyar" },
    { code: "iku", name: "Inuktitut", native: "ᐃᓄᒃᑎᑐᑦ" },
    { code: "ind", name: "Indonesian", native: "Bahasa Indonesia" },
    { code: "isl", name: "Icelandic", native: "Íslenska" },
    { code: "ita", name: "Italian", native: "italiano" },
    { code: "ita_old", name: "Italian Old", native: "italiano (Old)" },
    { code: "jav", name: "Javanese", native: "basa Jawa" },
    { code: "jpn", name: "Japanese", native: "日本語" },
    { code: "kan", name: "Kannada", native: "ಕನ್ನಡ" },
    { code: "kat", name: "Georgian", native: "ქართული" },
    { code: "kat_old", name: "Georgian Old", native: "ქართული (Old)" },
    { code: "kaz", name: "Kazakh", native: "қазақ тілі" },
    { code: "khm", name: "Khmer", native: "ភាសាខ្មែរ" },
    { code: "kir", name: "Kyrgyz", native: "кыргызча" },
    { code: "kor", name: "Korean", native: "한국어" },
    { code: "lao", name: "Lao", native: "ພາສາລາວ" },
    { code: "lat", name: "Latin", native: "latine" },
    { code: "lav", name: "Latvian", native: "latviešu valoda" },
    { code: "lit", name: "Lithuanian", native: "lietuvių kalba" },
    { code: "mal", name: "Malayalam", native: "മലയാളം" },
    { code: "mar", name: "Marathi", native: "मराठी" },
    { code: "mkd", name: "Macedonian", native: "македонски јазик" },
    { code: "mlt", name: "Maltese", native: "Malti" },
    { code: "mon", name: "Mongolian", native: "монгол" },
    { code: "msa", name: "Malay", native: "bahasa Melayu" },
    { code: "mya", name: "Myanmar", native: "ဗမာစာ" },
    { code: "nep", name: "Nepali", native: "नेपाली" },
    { code: "nld", name: "Dutch", native: "Nederlands" },
    { code: "nor", name: "Norwegian", native: "norsk" },
    { code: "ori", name: "Oriya", native: "ଓଡ଼ିଆ" },
    { code: "pan", name: "Punjabi", native: "ਪੰਜਾਬੀ" },
    { code: "pol", name: "Polish", native: "polski" },
    { code: "por", name: "Portuguese", native: "português" },
    { code: "pus", name: "Pashto", native: "پښتو" },
    { code: "ron", name: "Romanian", native: "română" },
    { code: "rus", name: "Russian", native: "русский язык" },
    { code: "san", name: "Sanskrit", native: "संस्कृतम्" },
    { code: "sin", name: "Sinhala", native: "සිංහල" },
    { code: "slk", name: "Slovak", native: "slovenčina" },
    { code: "slv", name: "Slovenian", native: "slovenščina" },
    { code: "spa", name: "Spanish", native: "español" },
    { code: "spa_old", name: "Spanish Old", native: "español (Old)" },
    { code: "sqi", name: "Albanian", native: "shqip" },
    { code: "srp", name: "Serbian", native: "српски језик" },
    { code: "srp_latn", name: "Serbian Latin", native: "srpski (latin)" },
    { code: "swa", name: "Swahili", native: "Kiswahili" },
    { code: "swe", name: "Swedish", native: "svenska" },
    { code: "syr", name: "Syriac", native: "ܠܫܢܐ ܣܘܪܝܝܐ" },
    { code: "tam", name: "Tamil", native: "தமிழ்" },
    { code: "tel", name: "Telugu", native: "తెలుగు" },
    { code: "tgk", name: "Tajik", native: "тоҷикӣ" },
    { code: "tgl", name: "Tagalog", native: "Wikang Tagalog" },
    { code: "tha", name: "Thai", native: "ไทย" },
    { code: "tir", name: "Tigrinya", native: "ትግርኛ" },
    { code: "tur", name: "Turkish", native: "Türkçe" },
    { code: "uig", name: "Uyghur", native: "ئۇيغۇرچە" },
    { code: "ukr", name: "Ukrainian", native: "українська мова" },
    { code: "urd", name: "Urdu", native: "اردو" },
    { code: "uzb", name: "Uzbek", native: "oʻzbek" },
    { code: "uzb_cyrl", name: "Uzbek Cyrillic", native: "ўзбек" },
    { code: "vie", name: "Vietnamese", native: "Tiếng Việt" },
    { code: "yid", name: "Yiddish", native: "ייִדיש" },
  ];
}

async function performHealthCheck(): Promise<any> {
  try {
    // Test Tesseract availability
    const rawBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
    Buffer.from(rawBase64, 'base64'); // sanity
    const pooled = await acquireWorker('eng');
    releaseWorker(pooled);

    return {
      status: "healthy",
      tesseract: "available",
      languages: getSupportedLanguages().length,
      features: ["OCR", "language_detection", "preprocessing"],
      timestamp: new Date().toISOString(),
    };
  } catch (error: any) {
    return {
      status: "degraded",
      tesseract: "unavailable",
      error: error.message,
      timestamp: new Date().toISOString(),
    };
  }
}

// Batch processing endpoint
export const PUT: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const { action, ...params } = await request.json();

    switch (action) {
      case "batch_ocr":
        const { imageUrls, languages = "eng", parallel = 2 } = params;

        if (!Array.isArray(imageUrls)) {
          return json({ error: "imageUrls must be an array" }, { status: 400 });
        }
        const limited = imageUrls.slice(0, 100); // hard cap safeguard
        const results: any[] = [];
        const start = Date.now();
        const pLimit = parallel > 0 ? parallel : 2;
        let idx = 0;
        async function next(): Promise<void> {
          if (idx >= limited.length) return;
          const current = idx++;
          const url = limited[current];
          try {
              const res = await fetch(url);
              const arrayBuf = await res.arrayBuffer();
              const buf = Buffer.from(arrayBuf);
              if (!looksLikeImage(buf)) throw new Error('BAD_IMAGE_MAGIC');
              const meta = { width: undefined, height: undefined };
              const ocr = await performOCR(buf, languages.split(',').join('+'), meta);
              results.push({ imageUrl: url, success: true, text: ocr.text, confidence: ocr.confidence });
            } catch (e: any) {
              results.push({ imageUrl: url, success: false, error: e.message });
            }
          await next();
        }
        // Launch limited parallel workers
        await Promise.all(Array.from({ length: Math.min(pLimit, limited.length) }, () => next()));
        const duration = Date.now() - start;
        return json({
          success: true,
          results,
          summary: {
            total: limited.length,
            successful: results.filter((r) => r.success).length,
            failed: results.filter((r) => !r.success).length,
            durationMs: duration,
          },
        });
      default:
        return json(
          { error: "Unknown action", availableActions: ["batch_ocr"] },
          { status: 400 }
        );
    }
  } catch (error: any) {
    console.error("OCR batch operation error:", error);
    return json(
      { error: "Batch operation failed", details: error.message },
      { status: 500 }
    );
  }
};
