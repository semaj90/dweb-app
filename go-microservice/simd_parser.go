package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	"github.com/minio/simdjson-go"
	"github.com/valyala/fastjson"
)

// SIMDParser handles high-performance JSON parsing with SIMD optimizations
type SIMDParser struct {
	simdjsonParser *simdjson.Parser
	fastjsonParser *fastjson.Parser
	sonicAPI       sonic.API
	mu             sync.RWMutex
	metrics        *ParserMetrics
	bufferPool     sync.Pool
}

// ParserMetrics tracks parsing performance
type ParserMetrics struct {
	mu                 sync.RWMutex
	TotalParses        int64         `json:"total_parses"`
	SimdParses         int64         `json:"simd_parses"`
	FastParses         int64         `json:"fast_parses"`
	SonicParses        int64         `json:"sonic_parses"`
	TotalBytes         int64         `json:"total_bytes"`
	TotalParseTime     time.Duration `json:"total_parse_time"`
	AvgParseTime       time.Duration `json:"avg_parse_time"`
	ParseErrors        int64         `json:"parse_errors"`
	LastUpdate         time.Time     `json:"last_update"`
	BytesPerSecond     float64       `json:"bytes_per_second"`
	DocumentsPerSecond float64       `json:"documents_per_second"`
}

// ParseResult represents the result of a parse operation
type ParseResult struct {
	Success        bool            `json:"success"`
	Data           interface{}     `json:"data"`
	ParseTime      time.Duration   `json:"parse_time"`
	BytesProcessed int64           `json:"bytes_processed"`
	ParserUsed     string          `json:"parser_used"`
	Error          string          `json:"error,omitempty"`
}

// ChunkProcessor handles streaming JSON chunk processing
type ChunkProcessor struct {
	parser      *SIMDParser
	bufferSize  int
	concurrency int
	results     chan *ParseResult
	errors      chan error
}

// JSONChunk represents a chunk of JSON data
type JSONChunk struct {
	ID        string          `json:"id"`
	Index     int             `json:"index"`
	Data      json.RawMessage `json:"data"`
	Size      int             `json:"size"`
	Timestamp time.Time       `json:"timestamp"`
}

// StreamConfig configuration for streaming parser
type StreamConfig struct {
	BufferSize      int           `json:"buffer_size"`
	MaxChunkSize    int           `json:"max_chunk_size"`
	Concurrency     int           `json:"concurrency"`
	UseCompression  bool          `json:"use_compression"`
	ParseStrategy   string        `json:"parse_strategy"` // "simd", "fast", "sonic", "auto"
	EnableMetrics   bool          `json:"enable_metrics"`
	TimeoutPerChunk time.Duration `json:"timeout_per_chunk"`
}

var (
	simdParser     *SIMDParser
	simdParserOnce sync.Once
)

// InitializeSIMDParser initializes the global SIMD parser
func InitializeSIMDParser() error {
	var initErr error
	simdParserOnce.Do(func() {
		simdParser = &SIMDParser{
			simdjsonParser: &simdjson.Parser{},
			fastjsonParser: &fastjson.Parser{},
			sonicAPI:       sonic.ConfigDefault,
			metrics:        &ParserMetrics{LastUpdate: time.Now()},
			bufferPool: sync.Pool{
				New: func() interface{} {
					return bytes.NewBuffer(make([]byte, 0, 1024*1024)) // 1MB buffer
				},
			},
		}

		// Start metrics updater
		go simdParser.updateMetrics()

		log.Printf("✅ SIMD JSON Parser initialized with %d CPU cores", runtime.NumCPU())
	})

	return initErr
}

// GetSIMDParser returns the global SIMD parser instance
func GetSIMDParser() *SIMDParser {
	return simdParser
}

// ParseJSON parses JSON using the most appropriate parser
func (sp *SIMDParser) ParseJSON(data []byte) (*ParseResult, error) {
	startTime := time.Now()
	dataSize := int64(len(data))
	
	// Choose parser based on data size and CPU capabilities
	var result interface{}
	var err error
	var parserUsed string

	// Try SIMD parser first for large documents
	if dataSize > 10*1024 { // > 10KB
		result, err = sp.parseSIMD(data)
		parserUsed = "simdjson"
		sp.updateParseMetrics("simd", dataSize, startTime)
	} else if dataSize > 1024 { // > 1KB
		result, err = sp.parseSonic(data)
		parserUsed = "sonic"
		sp.updateParseMetrics("sonic", dataSize, startTime)
	} else {
		result, err = sp.parseFast(data)
		parserUsed = "fastjson"
		sp.updateParseMetrics("fast", dataSize, startTime)
	}

	if err != nil {
		sp.metrics.mu.Lock()
		sp.metrics.ParseErrors++
		sp.metrics.mu.Unlock()
		
		return &ParseResult{
			Success:        false,
			Error:          err.Error(),
			ParseTime:      time.Since(startTime),
			BytesProcessed: dataSize,
			ParserUsed:     parserUsed,
		}, err
	}

	return &ParseResult{
		Success:        true,
		Data:           result,
		ParseTime:      time.Since(startTime),
		BytesProcessed: dataSize,
		ParserUsed:     parserUsed,
	}, nil
}

// parseSIMD uses simdjson for SIMD-accelerated parsing
func (sp *SIMDParser) parseSIMD(data []byte) (interface{}, error) {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	parsed, err := sp.simdjsonParser.Parse(data, nil)
	if err != nil {
		return nil, err
	}

	// Convert to Go types
	return sp.convertSIMDToGo(parsed)
}

// parseSonic uses bytedance/sonic for fast parsing
func (sp *SIMDParser) parseSonic(data []byte) (interface{}, error) {
	var result interface{}
	err := sp.sonicAPI.Unmarshal(data, &result)
	return result, err
}

// parseFast uses fastjson for small documents
func (sp *SIMDParser) parseFast(data []byte) (interface{}, error) {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	v, err := sp.fastjsonParser.ParseBytes(data)
	if err != nil {
		return nil, err
	}

	return convertFastJSONValue(v), nil
}

// convertSIMDToGo converts simdjson parsed data to Go types
func (sp *SIMDParser) convertSIMDToGo(parsed *simdjson.ParsedJson) (interface{}, error) {
	iter := parsed.Iter
	
	// Get the root element type
	t := iter.Type()
	
	switch t {
	case simdjson.TypeObject:
		return sp.parseObject(&iter)
	case simdjson.TypeArray:
		return sp.parseArray(&iter)
	case simdjson.TypeString:
		str, _ := iter.String()
		return str, nil
	case simdjson.TypeInt:
		val, _ := iter.Int()
		return val, nil
	case simdjson.TypeFloat:
		val, _ := iter.Float()
		return val, nil
	case simdjson.TypeBool:
		val, _ := iter.Bool()
		return val, nil
	case simdjson.TypeNull:
		return nil, nil
	default:
		return nil, fmt.Errorf("unknown type: %v", t)
	}
}

// parseObject parses a JSON object using simdjson
func (sp *SIMDParser) parseObject(iter *simdjson.Iter) (map[string]interface{}, error) {
	obj := make(map[string]interface{})
	
	// Iterate through object
	objIter, err := iter.Object(nil)
	if err != nil {
		return nil, err
	}
	
	for {
		key, t, err := objIter.NextElement(nil)
		if err != nil {
			break
		}
		
		switch t {
		case simdjson.TypeString:
			val, _ := objIter.Iter.StringCvt()
			obj[string(key)] = val
		case simdjson.TypeInt:
			val, _ := objIter.Iter.Int()
			obj[string(key)] = val
		case simdjson.TypeFloat:
			val, _ := objIter.Iter.Float()
			obj[string(key)] = val
		case simdjson.TypeBool:
			val, _ := objIter.Iter.Bool()
			obj[string(key)] = val
		case simdjson.TypeNull:
			obj[string(key)] = nil
		case simdjson.TypeObject:
			val, _ := sp.parseObject(&objIter.Iter)
			obj[string(key)] = val
		case simdjson.TypeArray:
			val, _ := sp.parseArray(&objIter.Iter)
			obj[string(key)] = val
		}
	}
	
	return obj, nil
}

// parseArray parses a JSON array using simdjson
func (sp *SIMDParser) parseArray(iter *simdjson.Iter) ([]interface{}, error) {
	arr := make([]interface{}, 0)
	
	arrayIter, err := iter.Array(nil)
	if err != nil {
		return nil, err
	}
	
	for {
		t, err := arrayIter.Iter.AdvanceIter()
		if err != nil {
			break
		}
		
		switch t {
		case simdjson.TypeString:
			val, _ := arrayIter.Iter.StringCvt()
			arr = append(arr, val)
		case simdjson.TypeInt:
			val, _ := arrayIter.Iter.Int()
			arr = append(arr, val)
		case simdjson.TypeFloat:
			val, _ := arrayIter.Iter.Float()
			arr = append(arr, val)
		case simdjson.TypeBool:
			val, _ := arrayIter.Iter.Bool()
			arr = append(arr, val)
		case simdjson.TypeNull:
			arr = append(arr, nil)
		case simdjson.TypeObject:
			val, _ := sp.parseObject(&arrayIter.Iter)
			arr = append(arr, val)
		case simdjson.TypeArray:
			val, _ := sp.parseArray(&arrayIter.Iter)
			arr = append(arr, val)
		}
	}
	
	return arr, nil
}

// ParseStream parses a stream of JSON data
func (sp *SIMDParser) ParseStream(reader io.Reader, config StreamConfig) (<-chan *ParseResult, <-chan error) {
	results := make(chan *ParseResult, config.Concurrency)
	errors := make(chan error, config.Concurrency)
	
	go func() {
		defer close(results)
		defer close(errors)
		
		// Create chunk processor
		processor := &ChunkProcessor{
			parser:      sp,
			bufferSize:  config.BufferSize,
			concurrency: config.Concurrency,
			results:     results,
			errors:      errors,
		}
		
		// Process stream in chunks
		processor.processStream(reader, config)
	}()
	
	return results, errors
}

// processStream processes a stream of JSON data in chunks
func (cp *ChunkProcessor) processStream(reader io.Reader, config StreamConfig) {
	// Create worker pool
	var wg sync.WaitGroup
	chunkChan := make(chan *JSONChunk, cp.concurrency)
	
	// Start workers
	for i := 0; i < cp.concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cp.processChunks(chunkChan, config)
		}()
	}
	
	// Read and distribute chunks
	buffer := make([]byte, config.BufferSize)
	chunkIndex := 0
	
	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			chunk := &JSONChunk{
				ID:        fmt.Sprintf("chunk_%d", chunkIndex),
				Index:     chunkIndex,
				Data:      json.RawMessage(buffer[:n]),
				Size:      n,
				Timestamp: time.Now(),
			}
			
			select {
			case chunkChan <- chunk:
				chunkIndex++
			case <-time.After(config.TimeoutPerChunk):
				cp.errors <- fmt.Errorf("timeout sending chunk %d", chunkIndex)
			}
		}
		
		if err == io.EOF {
			break
		} else if err != nil {
			cp.errors <- err
			break
		}
	}
	
	close(chunkChan)
	wg.Wait()
}

// processChunks processes individual chunks
func (cp *ChunkProcessor) processChunks(chunks <-chan *JSONChunk, config StreamConfig) {
	for chunk := range chunks {
		// Parse chunk based on strategy
		var result *ParseResult
		var err error
		
		switch config.ParseStrategy {
		case "simd":
			result, err = cp.parser.ParseJSON(chunk.Data)
		case "sonic":
			var data interface{}
			err = sonic.Unmarshal(chunk.Data, &data)
			result = &ParseResult{
				Success:        err == nil,
				Data:           data,
				BytesProcessed: int64(chunk.Size),
				ParserUsed:     "sonic",
			}
		case "fast":
			data, parseErr := cp.parser.parseFast(chunk.Data)
			result = &ParseResult{
				Success:        parseErr == nil,
				Data:           data,
				BytesProcessed: int64(chunk.Size),
				ParserUsed:     "fastjson",
			}
			err = parseErr
		default: // auto
			result, err = cp.parser.ParseJSON(chunk.Data)
		}
		
		if err != nil {
			cp.errors <- fmt.Errorf("error parsing chunk %d: %v", chunk.Index, err)
		} else if result != nil {
			cp.results <- result
		}
	}
}

// ParseBatch parses multiple JSON documents in parallel
func (sp *SIMDParser) ParseBatch(documents [][]byte) ([]*ParseResult, error) {
	results := make([]*ParseResult, len(documents))
	var wg sync.WaitGroup
	var mu sync.Mutex
	
	// Process documents in parallel
	for i, doc := range documents {
		wg.Add(1)
		go func(index int, data []byte) {
			defer wg.Done()
			
			result, err := sp.ParseJSON(data)
			if err != nil {
				result = &ParseResult{
					Success:        false,
					Error:          err.Error(),
					BytesProcessed: int64(len(data)),
				}
			}
			
			mu.Lock()
			results[index] = result
			mu.Unlock()
		}(i, doc)
	}
	
	wg.Wait()
	return results, nil
}

// ValidateJSON validates JSON data using SIMD
func (sp *SIMDParser) ValidateJSON(data []byte) (bool, error) {
	// Use simdjson for validation (very fast)
	_, err := sp.simdjsonParser.Parse(data, nil)
	return err == nil, err
}

// ExtractPath extracts a specific path from JSON using SIMD
func (sp *SIMDParser) ExtractPath(data []byte, path string) (interface{}, error) {
	// Parse the JSON
	parsed, err := sp.simdjsonParser.Parse(data, nil)
	if err != nil {
		return nil, err
	}
	
	// Navigate to path (simplified - in production, implement full JSONPath)
	iter := parsed.Iter
	pathSegments := bytes.Split([]byte(path), []byte("."))
	
	for _, segment := range pathSegments {
		if iter.Type() == simdjson.TypeObject {
			objIter, _ := iter.Object(nil)
			found := false
			
			for {
				key, _, err := objIter.NextElement(nil)
				if err != nil {
					break
				}
				
				if bytes.Equal(key, segment) {
					iter = objIter.Iter
					found = true
					break
				}
			}
			
			if !found {
				return nil, fmt.Errorf("path not found: %s", string(segment))
			}
		}
	}
	
	return sp.convertIterToGo(&iter)
}

// convertIterToGo converts a simdjson iterator to Go type
func (sp *SIMDParser) convertIterToGo(iter *simdjson.Iter) (interface{}, error) {
	switch iter.Type() {
	case simdjson.TypeString:
		return iter.StringCvt()
	case simdjson.TypeInt:
		return iter.Int()
	case simdjson.TypeFloat:
		return iter.Float()
	case simdjson.TypeBool:
		return iter.Bool()
	case simdjson.TypeNull:
		return nil, nil
	case simdjson.TypeObject:
		return sp.parseObject(iter)
	case simdjson.TypeArray:
		return sp.parseArray(iter)
	default:
		return nil, fmt.Errorf("unknown type")
	}
}

// GetMetrics returns current parser metrics
func (sp *SIMDParser) GetMetrics() *ParserMetrics {
	sp.metrics.mu.RLock()
	defer sp.metrics.mu.RUnlock()
	return sp.metrics
}

// updateParseMetrics updates parsing metrics
func (sp *SIMDParser) updateParseMetrics(parserType string, dataSize int64, startTime time.Time) {
	sp.metrics.mu.Lock()
	defer sp.metrics.mu.Unlock()
	
	sp.metrics.TotalParses++
	sp.metrics.TotalBytes += dataSize
	parseTime := time.Since(startTime)
	sp.metrics.TotalParseTime += parseTime
	
	switch parserType {
	case "simd":
		sp.metrics.SimdParses++
	case "fast":
		sp.metrics.FastParses++
	case "sonic":
		sp.metrics.SonicParses++
	}
	
	if sp.metrics.TotalParses > 0 {
		sp.metrics.AvgParseTime = sp.metrics.TotalParseTime / time.Duration(sp.metrics.TotalParses)
	}
	
	// Calculate throughput
	elapsed := time.Since(sp.metrics.LastUpdate).Seconds()
	if elapsed > 0 {
		sp.metrics.BytesPerSecond = float64(sp.metrics.TotalBytes) / elapsed
		sp.metrics.DocumentsPerSecond = float64(sp.metrics.TotalParses) / elapsed
	}
}

// updateMetrics updates metrics periodically
func (sp *SIMDParser) updateMetrics() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		sp.metrics.mu.Lock()
		sp.metrics.LastUpdate = time.Now()
		sp.metrics.mu.Unlock()
	}
}

// CompareJSONDocuments compares two JSON documents using SIMD
func (sp *SIMDParser) CompareJSONDocuments(doc1, doc2 []byte) (bool, error) {
	// Parse both documents
	parsed1, err := sp.simdjsonParser.Parse(doc1, nil)
	if err != nil {
		return false, err
	}
	
	parsed2, err := sp.simdjsonParser.Parse(doc2, nil)
	if err != nil {
		return false, err
	}
	
	// Convert to Go types and compare
	obj1, err := sp.convertSIMDToGo(parsed1)
	if err != nil {
		return false, err
	}
	
	obj2, err := sp.convertSIMDToGo(parsed2)
	if err != nil {
		return false, err
	}
	
	// Simple equality check (in production, implement deep comparison)
	return fmt.Sprintf("%v", obj1) == fmt.Sprintf("%v", obj2), nil
}

// MinifyJSON minifies JSON data using SIMD parsing
func (sp *SIMDParser) MinifyJSON(data []byte) ([]byte, error) {
	// Parse and re-encode without whitespace
	parsed, err := sp.ParseJSON(data)
	if err != nil {
		return nil, err
	}
	
	return sonic.Marshal(parsed.Data)
}

// PrettyJSON formats JSON data with indentation
func (sp *SIMDParser) PrettyJSON(data []byte) ([]byte, error) {
	// Parse and re-encode with indentation
	parsed, err := sp.ParseJSON(data)
	if err != nil {
		return nil, err
	}
	
	return sonic.MarshalIndent(parsed.Data, "", "  ")
}
