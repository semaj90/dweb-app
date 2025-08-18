package main

import (
	"fmt"

	"github.com/gin-gonic/gin"
)

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "parser": "simd-health"})
	})

	fmt.Println("SIMD Health server on :8080")
	_ = r.Run(":8080")
}
