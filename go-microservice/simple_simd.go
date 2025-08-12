//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"time"
	"github.com/gin-gonic/gin"
	"github.com/minio/simdjson-go"
)

type SimpleParser struct {
	parser *simdjson.Parser
}

func NewSimpleParser() *SimpleParser {
	p := simdjson.NewParser()
	p.SetCapacity(100 << 20) // 100MB
	return &SimpleParser{parser: p}
}

func (s *SimpleParser) Parse(data []byte) (map[string]interface{}, error) {
	pj, err := s.parser.Parse(data, nil)
	if err != nil {
		return nil, err
	}
	
	iter := pj.Iter()
	iter.Advance()
	
	// Extract basic metrics
	metrics := map[string]interface{}{
		"type":        iter.Type().String(),
		"tape_length": pj.Tape.Len(),
		"size":        len(data),
	}
	
	// Count top-level fields
	if iter.Type() == simdjson.TypeObject {
		count := 0
		obj, _ := iter.Object(nil)
		obj.Map(func(key []byte, iter simdjson.Iter) {
			count++
		})
		metrics["field_count"] = count
	}
	
	return metrics, nil
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	
	parser := NewSimpleParser()
	
	r.POST("/parse/simple", func(c *gin.Context) {
		data, _ := c.GetRawData()
		start := time.Now()
		
		result, err := parser.Parse(data)
		elapsed := time.Since(start)
		
		if err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		
		throughput := float64(len(data)) / elapsed.Seconds() / (1<<30) // GB/s
		
		c.JSON(200, gin.H{
			"parser":     "simdjson-go",
			"result":     result,
			"parse_ns":   elapsed.Nanoseconds(),
			"throughput": fmt.Sprintf("%.2f GB/s", throughput),
		})
	})
	
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "operational", "simd": true})
	})
	
	fmt.Println("Simple SIMD Parser (simdjson-go) on :8080")
	r.Run(":8080")
}
