package main

import (
	"log"
	"microservice/service"
)

func main() {
	if err := service.RunServer(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
