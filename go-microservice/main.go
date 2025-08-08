package main

import (
	"fmt"
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
		
		var fieldCount int
		if obj := parsed.GetObject(); obj != nil {
			obj.Visit(func(key []byte, v *fastjson.Value) {
				fieldCount++
			})
		}
		
		c.JSON(200, gin.H{
			"processed": true,
			"fields": fieldCount,
		})
	})
	
	r.POST("/index", func(c *gin.Context) {
		c.JSON(202, gin.H{"status": "indexing_started"})
	})
	
	fmt.Println("Server running on :8080")
	r.Run(":8080")
}
