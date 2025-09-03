package main

import (
	"fmt"
	"os"
	"strconv"
)

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if value, exists := os.LookupEnv(key); exists {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return fallback
}

func getEnvFloat(key string, fallback float64) float64 {
	if value, exists := os.LookupEnv(key); exists {
		if floatValue, err := strconv.ParseFloat(value, 64); err == nil {
			return floatValue
		}
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	if value, exists := os.LookupEnv(key); exists {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return fallback
}

func main() {
	fmt.Println("Environment Utility")
	fmt.Println("==================")
	
	// Display all environment variables
	for _, env := range os.Environ() {
		fmt.Println(env)
	}
	
	// Test environment variable functions
	fmt.Printf("DATABASE_URL: %s\n", getEnv("DATABASE_URL", "not set"))
	fmt.Printf("WORKER_COUNT: %d\n", getEnvInt("WORKER_COUNT", 4))
	fmt.Printf("DEBUG: %t\n", getEnvBool("DEBUG", false))
	fmt.Printf("TIMEOUT: %f\n", getEnvFloat("TIMEOUT", 30.0))
}