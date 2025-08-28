package main

import (
	"context"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"time"

	"github.com/go-redis/redis/v8"
	_ "github.com/jackc/pgx/v5/stdlib"
)

var ctx = context.Background()

// naive Qdrant upsert payload
type qdrantPoint struct {
	Id string `json:"id"`
	Vector []float32 `json:"vector"`
	Payload map[string]any `json:"payload"`
}

func processBatchToQdrant(points []qdrantPoint, qdrantUrl string) error {
	if len(points) == 0 { return nil }
	payload := map[string]interface{}{"points": points}
	b, _ := json.Marshal(payload)
	resp, err := http.Post(qdrantUrl+"/collections/feedback/points?wait=true", "application/json", bytesNewReader(b))
	if err != nil { return err }
	defer resp.Body.Close()
	_, _ = ioutil.ReadAll(resp.Body)
	return nil
}

func bytesNewReader(b []byte) *bytesReader { return &bytesReader{b: b, i:0} }

type bytesReader struct{ b []byte; i int }

func (r *bytesReader) Read(p []byte) (int, error) {
	if r.i >= len(r.b) { return 0, ioEOF }
	n := copy(p, r.b[r.i:])
	r.i += n
	return n, nil
}

var ioEOF = fmtError("EOF")

func fmtError(s string) error { return &simpleError{s} }

type simpleError struct{ s string }
func (e *simpleError) Error() string { return e.s }

func main() {
	// Connect to Redis stream and batch consume
	rdb := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	defer rdb.Close()

	qdrantUrl := "http://localhost:6333"

	for {
		// XREAD BLOCK 5000 STREAMS feedback:events 0
		res, err := rdb.XRead(ctx, &redis.XReadArgs{Streams: []string{"feedback:events", "0"}, Count: 100, Block: 5000}).Result()
		if err != nil {
			log.Println("xread error", err)
			time.Sleep(2 * time.Second)
			continue
		}
		var points []qdrantPoint
		for _, stream := range res {
			for _, msg := range stream.Messages {
				// attempt to parse proto from msg.Values["proto"]
				if protoRaw, ok := msg.Values["proto"].(string); ok {
					// In Redis streams the binary may be stored differently; here we assume base64 string or raw bytes
					_ = protoRaw
					// TODO: unmarshal and convert
				}
				// fallback dummy point
				points = append(points, qdrantPoint{Id: msg.ID, Vector: make([]float32, 128), Payload: map[string]any{"msg": msg.Values}})
			}
		}
		if err := processBatchToQdrant(points, qdrantUrl); err != nil {
			log.Println("qdrant upsert error", err)
		}
	}
}
