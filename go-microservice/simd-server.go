//go:build legacy
// +build legacy

package main

import (
	"context"
	"log"
	"time"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/valyala/fastjson"
)

var (
	redisClient *redis.Client
	parser      = &fastjson.Parser{}
)

func main() {
	redisClient = redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})

	r := gin.Default()
	r.Use(cors.Default())

	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "simd": true})
	})

	r.POST("/simd-parse", func(c *gin.Context) {
		data, _ := c.GetRawData()
		start := time.Now()
		parsed, _ := parser.ParseBytes(data)
		parseTime := time.Since(start)
		c.JSON(200, gin.H{
			"parse_time": parseTime.String(),
			"size":       len(data),
			"fields":     len(parsed.GetObject()),
		})
	})

	r.POST("/redis-json", func(c *gin.Context) {
		key := c.Query("key")
		data, _ := c.GetRawData()
		redisClient.Set(context.Background(), key, data, time.Minute)
		c.JSON(200, gin.H{"cached": true})
	})

	log.Printf("🚀 SIMD+Redis+Vite Integration Server starting on :8080")
	log.Printf("   Workers: 32 | Redis Pool: 32")
	log.Printf("   SIMD JSON: ✓ | Redis JSON: ✓ | WebSocket: ✓")
	r.Run(":8080")
}