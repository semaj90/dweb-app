package main

import (
	orchestrator "legal-ai-platform/internal/orchestrator"
	"log"
)

func main() {
	log.Println("[cmd/gpu-orchestrator] starting orchestrator")
	orchestrator.Run()
}
