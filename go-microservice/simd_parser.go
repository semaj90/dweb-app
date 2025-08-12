//go:build legacy
// +build legacy

package main

import (
	"fmt"
	"time"
	"github.com/gin-gonic/gin"
	"github.com/minio/simdjson-go"
)

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	
	parser := simdjson.NewParser()
	parser.SetCapacity(100 << 20) // 100MB
	
	r.POST("/parse/simd", func(c *gin.Context) {
		data, _ := c.GetRawData()
		start := time.Now()
		
		pj, err := parser.Parse(data, nil)
		if err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		
		iter := pj.Iter()
		iter.Advance()
		
		elapsed := time.Since(start)
		throughput := float64(len(data)) / elapsed.Seconds() / (1<<20) // MB/s
		
		c.JSON(200, gin.H{
			"parser": "simdjson-go",
			"size": len(data),
			"parse_time_ns": elapsed.Nanoseconds(),
			"throughput_mbps": throughput,
			"type": iter.Type().String(),
		})
	})
	
	r.POST("/parse/batch", func(c *gin.Context) {
		var req struct {
			Documents []string `json:"documents"`
		}
		c.ShouldBindJSON(&req)
		
		results := make([]interface{}, len(req.Documents))
		totalTime := int64(0)
		
		for i, doc := range req.Documents {
			start := time.Now()
			pj, err := parser.Parse([]byte(doc), nil)
			elapsed := time.Since(start).Nanoseconds()
			totalTime += elapsed
			
			if err != nil {
				results[i] = gin.H{"error": err.Error()}
			} else {
				iter := pj.Iter()
				iter.Advance()
				results[i] = gin.H{"type": iter.Type().String(), "time_ns": elapsed}
			}
		}
		
		c.JSON(200, gin.H{
			"batch_size": len(req.Documents),
			"total_time_ns": totalTime,
			"avg_time_ns": totalTime / int64(len(req.Documents)),
			"results": results,
		})
	})
	
	fmt.Println("SIMD JSON Parser (simdjson-go) on :8080")
	r.Run(":8080")
}
