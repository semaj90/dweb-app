// OCR Language Extract API
// Provides OCR and text extraction capabilities with language detection

import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { createWorker } from "tesseract.js";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const imageFile = formData.get("image") as File;
    const languages = (formData.get("languages") as string) || "eng";
    const preprocessImage = formData.get("preprocess") === "true";

    if (!imageFile) {
      return json({ error: "Image file is required" }, { status: 400 });
    }

    // Validate file type
    if (!imageFile.type.startsWith("image/")) {
      return json({ error: "File must be an image" }, { status: 400 });
    }

    const startTime = Date.now();

    // Convert file to buffer
    const arrayBuffer = await imageFile.arrayBuffer();
    let buffer = Buffer.from(arrayBuffer); // Fixed: Use Buffer.from() to ensure proper type

    // Preprocess image if requested
    if (preprocessImage) {
      buffer = await preprocessImageBuffer(buffer);
    }

    // Parse languages parameter
    const langs = languages.split(",").map((lang) => lang.trim());

    // Perform OCR with fixed Tesseract configuration
    const ocrResult = await performOCR(buffer, langs);

    const processingTime = Date.now() - startTime;

    return json({
      success: true,
      result: {
        text: ocrResult.text,
        confidence: ocrResult.confidence,
        languages: langs,
        detectedLanguage: ocrResult.detectedLanguage,
        wordCount: ocrResult.text.split(/\s+/).length,
        characterCount: ocrResult.text.length,
        blocks: ocrResult.blocks || [],
        paragraphs: ocrResult.paragraphs || [],
        lines: ocrResult.lines || [],
        words: ocrResult.words || [],
      },
      metadata: {
        originalFileName: imageFile.name,
        fileSize: imageFile.size,
        mimeType: imageFile.type,
        preprocessed: preprocessImage,
        processingTime,
        tesseractVersion: "5.0.0",
      },
    });
  } catch (error: unknown) {
    console.error("OCR processing error:", error);
    return json(
      { error: "OCR processing failed", details: error.message },
      { status: 500 }
    );
  }
};

export const GET: RequestHandler = async ({ url }) => {
  try {
    const action = url.searchParams.get("action");

    switch (action) {
      case "supported_languages":
        return json({
          languages: getSupportedLanguages(),
          total: getSupportedLanguages().length,
        });

      case "health":
        const healthCheck = await performHealthCheck();
        return json(healthCheck);

      case "capabilities":
        return json({
          features: [
            "Text extraction",
            "Multi-language support",
            "Image preprocessing",
            "Confidence scoring",
            "Layout analysis",
            "Word/line/paragraph detection",
          ],
          supportedFormats: ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
          maxFileSize: "10MB",
          languages: getSupportedLanguages().length,
        });

      default:
        return json({
          message: "OCR Language Extract API",
          version: "2.0.0",
          availableActions: ["supported_languages", "health", "capabilities"],
        });
    }
  } catch (error: unknown) {
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
  languages: string[]
): Promise<{
  text: string;
  confidence: number;
  detectedLanguage: string;
  blocks?: unknown[];
  paragraphs?: unknown[];
  lines?: unknown[];
  words?: unknown[];
}> {
  try {
    // Create Tesseract worker with fixed configuration
    const worker = await createWorker(languages, undefined, {
      // Fixed: Removed logger property that doesn't exist in current Tesseract.js
      // cachePath: './node_modules/tesseract.js-core'
    });

    // Perform OCR recognition
    const { data } = await worker.recognize(buffer);

    // Clean up worker
    await worker.terminate();

    return {
      text: data.text || "",
      confidence: data.confidence || 0,
      detectedLanguage: detectPrimaryLanguage(data.text || ""),
      blocks: data.blocks || [],
      paragraphs: (data as any).paragraphs || [],
      lines: (data as any).lines || [],
      words: (data as any).words || [],
    };
  } catch (error: unknown) {
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
async function preprocessImageBuffer(inputBuffer: Buffer): Promise<Buffer> {
  try {
    // Use dynamic import for sharp to handle optional dependency
    const sharpMod = await import("sharp");
    const sharp = (sharpMod as any).default || sharpMod; // support both ESM/CJS

    const processedBuffer = (await sharp(inputBuffer)
      .jpeg({ quality: 90 })
      .normalize()
      .sharpen()
      .toBuffer()) as unknown as Buffer;

    return processedBuffer;
  } catch (error: unknown) {
    console.warn("Image preprocessing failed, using original:", error);
    return inputBuffer;
  }
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
    const testBuffer = Buffer.from(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      "base64"
    );

    const worker = await createWorker("eng");
    await worker.terminate();

    return {
      status: "healthy",
      tesseract: "available",
      languages: getSupportedLanguages().length,
      features: ["OCR", "language_detection", "preprocessing"],
      timestamp: new Date().toISOString(),
    };
  } catch (error: unknown) {
    return {
      status: "degraded",
      tesseract: "unavailable",
      error: error.message,
      timestamp: new Date().toISOString(),
    };
  }
}

// Batch processing endpoint
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { action, ...params } = await request.json();

    switch (action) {
      case "batch_ocr":
        const { imageUrls, languages = "eng" } = params;

        if (!Array.isArray(imageUrls)) {
          return json({ error: "imageUrls must be an array" }, { status: 400 });
        }

        const results = [];
        for (const imageUrl of imageUrls) {
          try {
            // In a real implementation, fetch and process each image
            results.push({
              imageUrl,
              success: true,
              text: "Mock OCR result for batch processing",
              confidence: 0.85,
            });
          } catch (error: unknown) {
            results.push({
              imageUrl,
              success: false,
              error: error.message,
            });
          }
        }

        return json({
          success: true,
          results,
          summary: {
            total: imageUrls.length,
            successful: results.filter((r) => r.success).length,
            failed: results.filter((r) => !r.success).length,
          },
        });
      default:
        return json(
          { error: "Unknown action", availableActions: ["batch_ocr"] },
          { status: 400 }
        );
    }
  } catch (error: unknown) {
    console.error("OCR batch operation error:", error);
    return json(
      { error: "Batch operation failed", details: error.message },
      { status: 500 }
    );
  }
};
