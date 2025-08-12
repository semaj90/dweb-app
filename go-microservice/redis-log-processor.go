//go:build legacy
// +build legacy

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"time"
)

// LogEntry represents the structure of a Redis log entry
type LogEntry struct {
	Timestamp string `json:"timestamp"`
	Level     string `json:"level"`
	Message   string `json:"message"`
}

// parseLogLine parses a single Redis log line and returns a LogEntry
func parseLogLine(line string) (*LogEntry, error) {
	// Example Redis log format: [PID] DD Mon HH:MM:SS.mmm level message
	// Using a regex to capture timestamp, level, and message
	re := regexp.MustCompile(`^[\[]\d+[\]] (\d{2} \w{3} \d{2}:\d{2}:\d{2}\.\d{3}) (\w+) (.*)$`)
	matches := re.FindStringSubmatch(line)

	if len(matches) != 4 {
		return nil, fmt.Errorf("failed to parse log line: %s", line)
	}

	// Parse the timestamp
	// Redis log timestamp format: "DD Mon HH:MM:SS.mmm" (e.g., "06 Aug 15:30:00.123")
	// Go's time.Parse requires a reference time.
	// We'll use a dummy year for parsing, as Redis logs don't include it.
	// The actual year will be the current year.
	currentYear := time.Now().Year()
	timestampStr := fmt.Sprintf("%s %d", matches[1], currentYear)
	
	// Adjust the layout to match the Redis log format including milliseconds
	layout := "02 Jan 15:04:05.000 2006"
	
t, err := time.Parse(layout, timestampStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse timestamp '%s': %w", timestampStr, err)
	}

	return &LogEntry{
		Timestamp: t.Format(time.RFC3339Nano), // ISO 8601 format with nanoseconds
		Level:     matches[2],
		Message:   matches[3],
	}, nil
}

// sendLogToNodeJS sends the parsed log entry as JSON to the Node.js endpoint
func sendLogToNodeJS(logEntry *LogEntry, endpoint string) error {
	jsonData, err := json.Marshal(logEntry)
	if err != nil {
		return fmt.Errorf("failed to marshal log entry to JSON: %w", err)
	}

	resp, err := http.Post(endpoint, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to send log to Node.js: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Node.js endpoint returned non-OK status: %s", resp.Status)
	}

	return nil
}

func main() {
	logFilePath := "..\\redis-windows\\redis.log" // Path to Redis log file
	nodeJSEndpoint := "http://localhost:5173/api/log" // Node.js endpoint for Redis logs

	file, err := os.OpenFile(logFilePath, os.O_RDONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Fatalf("Failed to open Redis log file: %v", err)
	}
	defer file.Close()

	// Seek to the end of the file to start tailing new entries
	_, err = file.Seek(0, io.SeekEnd)
	if err != nil {
		log.Fatalf("Failed to seek to end of file: %v", err)
	}

	reader := bufio.NewReader(file)

	log.Printf("Tailing Redis log file: %s", logFilePath)
	log.Printf("Sending logs to Node.js endpoint: %s", nodeJSEndpoint)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				// Wait for new data
				time.Sleep(500 * time.Millisecond)
				continue
			}
			log.Printf("Error reading log file: %v", err)
			time.Sleep(1 * time.Second) // Wait before retrying
			continue
		}

		logEntry, parseErr := parseLogLine(line)
		if parseErr != nil {
			log.Printf("Warning: %v", parseErr)
			continue
		}

		sendErr := sendLogToNodeJS(logEntry, nodeJSEndpoint)
		if sendErr != nil {
			log.Printf("Error sending log to Node.js: %v", sendErr)
		} else {
			log.Printf("Sent log: %+v", logEntry)
		}
	}
}
