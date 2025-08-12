//go:build legacy
// +build legacy

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	
	"github.com/gin-gonic/gin"
	"github.com/valyala/fastjson"
)

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	
	parser := &fastjson.Parser{}
	
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status": "ok",
			"gpu": false, // Bypass CUDA for now
			"cpu_cores": runtime.NumCPU(),
		})
	})
	
	r.POST("/process-document", func(c *gin.Context) {
		data, _ := c.GetRawData()
		parsed, err := parser.ParseBytes(data)
		if err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, gin.H{
			"processed": true,
			"fields": len(parsed.GetObject()),
		})
	})
	
	r.POST("/index", func(c *gin.Context) {
		var req map[string]interface{}
		json.NewDecoder(c.Request.Body).Decode(&req)
		
		// Mock indexing
		c.JSON(202, gin.H{
			"status": "indexing_started",
			"path": req["rootPath"],
		})
	})
	
	fmt.Println("Server running on :8080")
	r.Run(":8080")
}
