package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

const (
	OLLAMA_API_URL        = "http://localhost:11434/api" // Default Ollama API URL
	SVELTEKIT_STORE_URL   = "http://localhost:5173/api/indexing/store"
	GO_SERVER_PORT        = ":8081"
)

// FileProcessRequest represents the incoming request from SvelteKit
type FileProcessRequest struct {
	Files []string `json:"files"`
}

// OllamaEmbeddingRequest for nomic-embed-text
type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// OllamaEmbeddingResponse
type OllamaEmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

// OllamaCompletionRequest for gemma3-legal
type OllamaCompletionRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// OllamaCompletionResponse
type OllamaCompletionResponse struct {
	Response string `json:"response"`
}

// ProcessedFile represents the data to be sent back to SvelteKit
type ProcessedFile struct {
	FilePath    string      `json:"filePath"`
	Content     string      `json:"content"`
	Embedding   []float64   `json:"embedding"`
	Summary     string      `json:"summary"`
	Metadata    interface{} `json:"metadata"` // For additional file metadata
}

func main() {
	http.HandleFunc("/process-files", processFilesHandler)
	log.Printf("Go file processor listening on port %s", GO_SERVER_PORT)
	log.Fatal(http.ListenAndServe(GO_SERVER_PORT, nil))
}

func processFilesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST requests are allowed", http.StatusMethodNotAllowed)
		return
	}

	var req FileProcessRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	var wg sync.WaitGroup
	resultsChan := make(chan ProcessedFile, len(req.Files))
	errorChan := make(chan error, len(req.Files))

	for _, filePath := range req.Files {
		wg.Add(1)
		go func(fp string) {
			defer wg.Done()
			processedFile, err := processSingleFile(fp)
			if err != nil {
				errorChan <- fmt.Errorf("failed to process file %s: %v", fp, err)
				return
			}
			resultsChan <- processedFile
		}(filePath)
	}

	wg.Wait()
	close(resultsChan)
	close(errorChan)

	// Collect errors
	var errors []string
	for err := range errorChan {
		errors = append(errors, err.Error())
	}

	// Send results to SvelteKit storage endpoint
	var processedFiles []ProcessedFile
	for pf := range resultsChan {
		processedFiles = append(processedFiles, pf)
	}

	if len(processedFiles) > 0 {
		if err := sendToSvelteKitStorage(processedFiles); err != nil {
			log.Printf("Error sending processed files to SvelteKit storage: %v", err)
			http.Error(w, fmt.Sprintf("Failed to send processed files to SvelteKit: %v", err), http.StatusInternalServerError)
			return
		}
	}

	if len(errors) > 0 {
		log.Printf("Processed with errors: %v", errors)
		w.WriteHeader(http.StatusPartialContent) // Some files failed
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "partial_success",
			"message": "Some files failed to process",
			"errors":  errors,
		})
		return
	}

	json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "All files processed and sent to SvelteKit"})
}

func processSingleFile(filePath string) (ProcessedFile, error) {
	log.Printf("Processing file: %s", filePath)

	contentBytes, err := os.ReadFile(filePath)
	if err != nil {
		return ProcessedFile{}, fmt.Errorf("error reading file %s: %v", filePath, err)
	}
	content := string(contentBytes)

	// Get Embedding
	embedding, err := getOllamaEmbedding(content)
	if err != nil {
		return ProcessedFile{}, fmt.Errorf("error getting embedding for %s: %v", filePath, err)
	}

	// Get Summary
	summary, err := getOllamaSummary(content)
	if err != nil {
		return ProcessedFile{}, fmt.Errorf("error getting summary for %s: %v", filePath, err)
	}

	// Get file metadata (optional)
	fileInfo, err := os.Stat(filePath)
	var metadata map[string]interface{}
	if err == nil {
		metadata = map[string]interface{}{
			"size":        fileInfo.Size(),
			"modTime":     fileInfo.ModTime().Format(time.RFC3339),
			"isDirectory": fileInfo.IsDir(),
			"mode":        fileInfo.Mode().String(),
		}
	} else {
		log.Printf("Warning: Could not get file info for %s: %v", filePath, err)
	}

	return ProcessedFile{
		FilePath:    filePath,
		Content:     content,
		Embedding:   embedding,
		Summary:     summary,
		Metadata:    metadata,
	},
nil
}

func getOllamaEmbedding(text string) ([]float64, error) {
	reqBody, err := json.Marshal(OllamaEmbeddingRequest{
		Model:  "nomic-embed-text",
		Prompt: text,
	})
	if err != nil {
		return nil, fmt.Errorf("error marshalling embedding request: %v", err)
	}

	resp, err := http.Post(OLLAMA_API_URL+"/embeddings", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("error sending embedding request to Ollama: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embedding API returned non-OK status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var res OllamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("error decoding embedding response: %v", err)
	}
	return res.Embedding, nil
}

func getOllamaSummary(text string) (string, error) {
	reqBody, err := json.Marshal(OllamaCompletionRequest{
		Model:  "gemma3-legal", // Assuming this model is available and fine-tuned for legal summaries
		Prompt: fmt.Sprintf("Summarize the following document, focusing on key legal aspects and important information:\n\n%s", text),
	})
	if err != nil {
		return "", fmt.Errorf("error marshalling summary request: %v", err)
	}

	resp, err := http.Post(OLLAMA_API_URL+"/generate", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("error sending summary request to Ollama: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama summary API returned non-OK status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Ollama /generate endpoint streams responses, so we need to read line by line
	// and concatenate the 'response' field from each JSON object.
	decoder := json.NewDecoder(resp.Body)
	var fullSummary string
	for {
		var res OllamaCompletionResponse
		if err := decoder.Decode(&res); err == io.EOF {
			break
		} else if err != nil {
			return "", fmt.Errorf("error decoding summary stream response: %v", err)
		}
		fullSummary += res.Response
	}

	return fullSummary, nil
}

func sendToSvelteKitStorage(processedFiles []ProcessedFile) error {
	reqBody, err := json.Marshal(processedFiles)
	if err != nil {
		return fmt.Errorf("error marshalling processed files for SvelteKit: %v", err)
	}

	resp, err := http.Post(SVELTEKIT_STORE_URL, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("error sending processed files to SvelteKit storage: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("svelteKit storage API returned non-OK status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	log.Println("Successfully sent processed files to SvelteKit storage.")
	return nil
}
