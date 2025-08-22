package main

import (
	"log"
	"legal-ai-production/internal/service"
)

func main() {
	if err := service.RunServer(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
