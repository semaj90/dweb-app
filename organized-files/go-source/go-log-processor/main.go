package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/hpcloud/tail"
)

const (
	logFilePath     = "/var/log/redis/redis.log" // Path to your Redis log file
	svelteKitAPIURL = "http://localhost:5173/api/log" // Your SvelteKit API endpoint
)

type LogEntry struct {
	Message string `json:"message"`
	// Add other fields if you parse more from the log line
}

func main() {
	// Ensure the log file exists before tailing
	if _, err := os.Stat(logFilePath); os.IsNotExist(err) {
		log.Fatalf("Log file not found: %s. Please ensure Redis logging is configured and the file exists.", logFilePath)
	}

	t, err := tail.TailFile(logFilePath, tail.Config{
		Follow:    true, // Follow new lines
		ReOpen:    true, // Reopen if the file is truncated or rotated
		MustExist: true, // Fail if the file doesn't exist initially
		Poll:      true, // Use polling instead of inotify for broader compatibility
	})
	if err != nil {
		log.Fatalf("Failed to tail log file: %v", err)
	}
	defer t.Cleanup()

	log.Printf("Tailing log file: %s", logFilePath)

	for line := range t.Lines {
		log.Printf("Received log line: %s", line.Text)

		// For simplicity, we're just sending the raw log line as the message.
		// In a real scenario, you'd parse this line into structured data.
		logEntry := LogEntry{
			Message: line.Text,
		}

		jsonData, err := json.Marshal(logEntry)
		if err != nil {
			log.Printf("Error marshalling JSON: %v", err)
			continue
		}

		req, err := http.NewRequest("POST", svelteKitAPIURL, bytes.NewBuffer(jsonData))
		if err != nil {
			log.Printf("Error creating HTTP request: %v", err)
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Error sending log to SvelteKit API: %v", err)
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			log.Printf("SvelteKit API returned non-OK status: %d, Body: %s", resp.StatusCode, string(bodyBytes))
		} else {
			log.Println("Log successfully sent to SvelteKit API")
		}
	}

	if t.Err() != nil {
		log.Printf("Error while tailing file: %v", t.Err())
	}
}
