// OCR Service for Legal Document Processing
import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { createWorker } from 'tesseract.js';

const app = express();
const PORT = process.env.OCR_PORT || 8601;

// Configure CORS and middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff', 'application/pdf'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only images and PDFs are allowed.'), false);
    }
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    service: 'ocr-service',
    port: PORT,
    timestamp: new Date().toISOString(),
    features: ['tesseract-ocr', 'image-processing', 'pdf-support']
  });
});

// OCR endpoint
app.post('/api/ocr/extract', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ 
        error: 'No file uploaded',
        message: 'Please provide a file to process'
      });
    }

    console.log(`📄 Processing file: ${req.file.originalname} (${req.file.size} bytes)`);
    
    // Create Tesseract worker
    const worker = await createWorker('eng');
    
    try {
      // Perform OCR on the uploaded file
      const { data } = await worker.recognize(req.file.buffer);
      
      // Extract useful information
      const result = {
        success: true,
        filename: req.file.originalname,
        fileSize: req.file.size,
        mimeType: req.file.mimetype,
        text: data.text,
        confidence: data.confidence,
        processingTime: Date.now(),
        wordCount: data.text.split(/\s+/).filter(word => word.length > 0).length,
        metadata: {
          blocks: data.blocks?.length || 0,
          paragraphs: data.paragraphs?.length || 0,
          lines: data.lines?.length || 0,
          words: data.words?.length || 0,
          symbols: data.symbols?.length || 0
        }
      };

      console.log(`✅ OCR completed: ${result.wordCount} words extracted`);
      res.json(result);

    } finally {
      await worker.terminate();
    }

  } catch (error) {
    console.error('❌ OCR processing failed:', error);
    res.status(500).json({
      error: 'OCR processing failed',
      message: error.message,
      filename: req.file?.originalname
    });
  }
});

// Batch OCR endpoint
app.post('/api/ocr/batch', upload.array('files', 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ 
        error: 'No files uploaded',
        message: 'Please provide files to process'
      });
    }

    console.log(`📄 Processing ${req.files.length} files`);
    
    const results = [];
    const worker = await createWorker('eng');
    
    try {
      for (const file of req.files) {
        console.log(`Processing: ${file.originalname}`);
        const { data } = await worker.recognize(file.buffer);
        
        results.push({
          filename: file.originalname,
          fileSize: file.size,
          text: data.text,
          confidence: data.confidence,
          wordCount: data.text.split(/\s+/).filter(word => word.length > 0).length
        });
      }

      res.json({
        success: true,
        fileCount: req.files.length,
        results: results,
        totalWords: results.reduce((sum, r) => sum + r.wordCount, 0)
      });

    } finally {
      await worker.terminate();
    }

  } catch (error) {
    console.error('❌ Batch OCR processing failed:', error);
    res.status(500).json({
      error: 'Batch OCR processing failed',
      message: error.message
    });
  }
});

// Get OCR status
app.get('/api/ocr/status', (req, res) => {
  res.json({
    service: 'OCR Service',
    status: 'operational',
    version: '1.0.0',
    engine: 'tesseract.js',
    supportedFormats: ['jpeg', 'png', 'gif', 'bmp', 'tiff', 'pdf'],
    maxFileSize: '50MB',
    features: [
      'Text extraction',
      'Confidence scoring',
      'Batch processing',
      'Multiple image formats',
      'Metadata extraction'
    ]
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ 
        error: 'File too large',
        message: 'File size exceeds 50MB limit'
      });
    }
  }
  
  console.error('Server error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🔍 OCR Service running on port ${PORT}`);
  console.log(`📋 Health check: http://localhost:${PORT}/health`);
  console.log(`📄 OCR endpoint: http://localhost:${PORT}/api/ocr/extract`);
  console.log(`📊 Status endpoint: http://localhost:${PORT}/api/ocr/status`);
});