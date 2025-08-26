package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Config for the worker
type Config struct {
	RedisURL      string
	RedisStream   string
	RedisGroup    string
	ConsumerName  string
	CudaWorkerExe string
	QdrantURL     string
	QdrantCol     string
	PGConn        string
	MaxRetries    int
}

type Worker struct {
	cfg     Config
	pool    *pgxpool.Pool
	redis   *redis.Client
	qclient *QdrantClient
	wg      sync.WaitGroup
	stopCh  chan struct{}
}

// VecRequest is the expected job payload shape coming from the Redis stream
type VecRequest struct {
	OutboxID  string                 `json:"outbox_id,omitempty"`
	OwnerType string                 `json:"owner_type"` // 'evidence' | 'report' | 'chunk' | 'document'
	OwnerID   string                 `json:"owner_id"`   // uuid as string
	Event     string                 `json:"event"`      // upsert | reembed | rotate
	Payload   map[string]interface{} `json:"payload,omitempty"`
	// For rotate jobs:
	Quat   *struct{ W, X, Y, Z float32 } `json:"quat,omitempty"`
	Points []float32                     `json:"points,omitempty"`
	// For embed jobs:
	Texts []string `json:"texts,omitempty"`
}

func NewWorker(cfg Config, pool *pgxpool.Pool) (*Worker, error) {
	opt, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		return nil, fmt.Errorf("parse redis url: %w", err)
	}
	redisClient := redis.NewClient(opt)
	
	// validate redis
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}

	qc := NewQdrantClient(cfg.QdrantURL, cfg.QdrantCol)

	w := &Worker{
		cfg:     cfg,
		pool:    pool,
		redis:   redisClient,
		qclient: qc,
		stopCh:  make(chan struct{}),
	}
	return w, nil
}

func (w *Worker) Close() {
	_ = w.redis.Close()
	w.qclient.Close()
}

func (w *Worker) Stop() {
	close(w.stopCh)
}

// Run starts the consumer loop
func (w *Worker) Run(ctx context.Context) error {
	// ensure group exists (mkstream)
	_ = w.redis.XGroupCreateMkStream(ctx, w.cfg.RedisStream, w.cfg.RedisGroup, "$").Err()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-w.stopCh:
			return nil
		default:
		}

		res, err := w.redis.XReadGroup(ctx, &redis.XReadGroupArgs{
			Group:    w.cfg.RedisGroup,
			Consumer: w.cfg.ConsumerName,
			Streams:  []string{w.cfg.RedisStream, ">"},
			Count:    1,
			Block:    5000 * time.Millisecond,
		}).Result()

		if err != nil {
			if errors.Is(err, context.Canceled) {
				return nil
			}
			if err == redis.Nil {
				continue
			}
			log.Printf("xreadgroup error: %v", err)
			time.Sleep(1 * time.Second)
			continue
		}

		for _, stream := range res {
			for _, msg := range stream.Messages {
				w.wg.Add(1)
				go func(msg redis.XMessage) {
					defer w.wg.Done()
					if err := w.processMessage(ctx, msg); err != nil {
						log.Printf("processMessage error id=%s: %v", msg.ID, err)
					}
				}(msg)
			}
		}
	}
}

// processMessage parses the Redis message and routes to worker logic
func (w *Worker) processMessage(ctx context.Context, msg redis.XMessage) error {
	var job VecRequest
	
	if raw, ok := msg.Values["payload"]; ok {
		// payload is stored as stringified JSON
		if s, ok := raw.(string); ok {
			if err := json.Unmarshal([]byte(s), &job); err != nil {
				return fmt.Errorf("unmarshal payload json: %w", err)
			}
		}
	} else {
		// compose JSON from fields
		m := make(map[string]interface{}, len(msg.Values))
		for k, v := range msg.Values {
			if sv, ok := v.(string); ok {
				// try parse JSON values
				var maybe interface{}
				if (strings.HasPrefix(sv, "{") || strings.HasPrefix(sv, "[")) {
					if json.Unmarshal([]byte(sv), &maybe) == nil {
						m[k] = maybe
						continue
					}
				}
				// try integers
				if i, err := strconv.Atoi(sv); err == nil {
					m[k] = i
					continue
				}
				m[k] = sv
			} else {
				m[k] = v
			}
		}
		b, _ := json.Marshal(m)
		if err := json.Unmarshal(b, &job); err != nil {
			return fmt.Errorf("unmarshal composite msg: %w (raw=%s)", err, string(b))
		}
	}

	// Basic validation
	if job.OwnerType == "" || job.OwnerID == "" {
		return fmt.Errorf("invalid job missing owner_type/owner_id (msgID=%s)", msg.ID)
	}

	// Route by event type
	switch job.Event {
	case "rotate":
		if job.Quat == nil || len(job.Points) == 0 {
			return fmt.Errorf("rotate job missing quat/points")
		}
		rotated, err := w.runCudaRotate(ctx, job)
		if err != nil {
			_ = w.bumpOutboxAttempt(ctx, job.OutboxID)
			return fmt.Errorf("cuda rotate failed: %w", err)
		}

		if err := w.saveRotatedPoints(ctx, job, rotated); err != nil {
			return fmt.Errorf("saveRotatedPoints: %w", err)
		}

		// Acknowledge
		if err := w.redis.XAck(ctx, w.cfg.RedisStream, w.cfg.RedisGroup, msg.ID).Err(); err != nil {
			log.Printf("XAck error: %v", err)
		}
		_ = w.redis.XDel(ctx, w.cfg.RedisStream, msg.ID).Err()
		return nil

	case "upsert", "reembed":
		resp, err := w.runCudaEmbed(ctx, job)
		if err != nil {
			_ = w.bumpOutboxAttempt(ctx, job.OutboxID)
			return fmt.Errorf("embed worker failed: %w", err)
		}

		// Extract vector from response
		vec, ok := resp["embedding"]
		if !ok {
			vec, ok = resp["vector"]
		}
		if !ok {
			return fmt.Errorf("embed response missing embedding")
		}

		embedding, err := normalizeNumericArray(vec)
		if err != nil {
			return fmt.Errorf("normalize embedding: %w", err)
		}

		// Update Postgres and Qdrant
		if err := w.upsertVectorAndOutbox(ctx, job, embedding); err != nil {
			return fmt.Errorf("upsertVectorAndOutbox: %w", err)
		}

		if err := w.qclient.UpsertPoint(job.OwnerID, embedding, job.Payload); err != nil {
			log.Printf("qdrant upsert error owner=%s: %v", job.OwnerID, err)
		}

		// Acknowledge
		if err := w.redis.XAck(ctx, w.cfg.RedisStream, w.cfg.RedisGroup, msg.ID).Err(); err != nil {
			log.Printf("XAck error: %v", err)
		}
		_ = w.redis.XDel(ctx, w.cfg.RedisStream, msg.ID).Err()
		return nil

	default:
		return fmt.Errorf("unknown event type %q", job.Event)
	}
}

// normalizeNumericArray converts interface to []float32
func normalizeNumericArray(v interface{}) ([]float32, error) {
	switch a := v.(type) {
	case []interface{}:
		out := make([]float32, 0, len(a))
		for _, it := range a {
			switch n := it.(type) {
			case float64:
				out = append(out, float32(n))
			case float32:
				out = append(out, n)
			case int:
				out = append(out, float32(n))
			case json.Number:
				f, _ := n.Float64()
				out = append(out, float32(f))
			default:
				return nil, fmt.Errorf("unsupported numeric type %T", it)
			}
		}
		return out, nil
	case []float64:
		out := make([]float32, len(a))
		for i, f := range a {
			out[i] = float32(f)
		}
		return out, nil
	case []float32:
		return a, nil
	default:
		return nil, fmt.Errorf("unsupported vector type %T", v)
	}
}

// bumpOutboxAttempt increments attempts counter
func (w *Worker) bumpOutboxAttempt(ctx context.Context, outboxID string) error {
	if outboxID == "" {
		return nil
	}
	_, err := w.pool.Exec(ctx, `UPDATE vector_outbox SET attempts = attempts + 1 WHERE id = $1`, outboxID)
	return err
}

// upsertVectorAndOutbox updates vectors table and marks outbox processed
func (w *Worker) upsertVectorAndOutbox(ctx context.Context, job VecRequest, embedding []float32) error {
	// Convert to []float64 for pgvector
	d := make([]float64, len(embedding))
	for i, v := range embedding {
		d[i] = float64(v)
	}
	
	tagPayload := job.Payload
	if tagPayload == nil {
		tagPayload = map[string]interface{}{"source": "cuda-worker"}
	}

	tx, err := w.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// Update existing vector row
	commandTag, err := tx.Exec(ctx,
		`UPDATE vectors SET embedding = $1, payload = $2::jsonb, updated_at = now()
         WHERE owner_type = $3 AND owner_id = $4`,
		d, toJSONB(tagPayload), job.OwnerType, job.OwnerID)
	if err != nil {
		return err
	}
	
	if commandTag.RowsAffected() == 0 {
		// Insert fallback
		_, err = tx.Exec(ctx,
			`INSERT INTO vectors (owner_type, owner_id, embedding, payload, created_at, updated_at)
             VALUES ($1,$2,$3,$4, now(), now())`, 
			job.OwnerType, job.OwnerID, d, toJSONB(tagPayload))
		if err != nil {
			return err
		}
	}

	// Mark outbox processed
	if job.OutboxID != "" {
		_, err = tx.Exec(ctx, `UPDATE vector_outbox SET processed_at = now() WHERE id = $1`, job.OutboxID)
		if err != nil {
			return err
		}
	}

	return tx.Commit(ctx)
}

func toJSONB(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// saveRotatedPoints stores rotated points back to DB
func (w *Worker) saveRotatedPoints(ctx context.Context, job VecRequest, rotated []float32) error {
	payload := map[string]interface{}{"rotated": rotated}
	_, err := w.pool.Exec(ctx, 
		`UPDATE evidence SET rotated_points = $1::jsonb WHERE id = $2`, 
		toJSONB(payload), job.OwnerID)
	return err
}

// runCudaRotate spawns CUDA worker for point rotation
func (w *Worker) runCudaRotate(ctx context.Context, job VecRequest) ([]float32, error) {
	in := map[string]interface{}{
		"jobId":  job.OutboxID,
		"type":   "rotate",
		"quat":   job.Quat,
		"points": job.Points,
	}
	b, _ := json.Marshal(in)

	cmd := exec.CommandContext(ctx, w.cfg.CudaWorkerExe)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, _ := cmd.StderrPipe()

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	// Send JSON input
	_, err = io.Copy(stdin, bytes.NewReader(b))
	stdin.Close()
	if err != nil {
		_ = cmd.Process.Kill()
		return nil, err
	}

	// Read output
	outBytes, err := io.ReadAll(stdout)
	if err != nil {
		_ = cmd.Process.Kill()
		return nil, err
	}
	
	if se, _ := io.ReadAll(stderr); len(se) > 0 {
		log.Printf("cuda-worker stderr: %s", string(se))
	}

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("cuda process error: %w; out=%s", err, string(outBytes))
	}

	// Parse response
	var resp map[string]interface{}
	if err := json.Unmarshal(outBytes, &resp); err != nil {
		return nil, fmt.Errorf("parse worker json: %w (raw=%s)", err, string(outBytes))
	}

	rawRot, ok := resp["rotated"]
	if !ok {
		return nil, fmt.Errorf("worker response missing 'rotated' field")
	}
	
	rot, err := normalizeNumericArray(rawRot)
	if err != nil {
		return nil, fmt.Errorf("normalize rotated: %w", err)
	}
	return rot, nil
}

// runCudaEmbed calls CUDA worker for embedding generation
func (w *Worker) runCudaEmbed(ctx context.Context, job VecRequest) (map[string]interface{}, error) {
	in := map[string]interface{}{
		"jobId":   job.OutboxID,
		"type":    "embed",
		"texts":   job.Texts,
		"payload": job.Payload,
	}
	b, _ := json.Marshal(in)

	cmd := exec.CommandContext(ctx, w.cfg.CudaWorkerExe)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, _ := cmd.StderrPipe()

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	_, err = io.Copy(stdin, bytes.NewReader(b))
	stdin.Close()
	if err != nil {
		_ = cmd.Process.Kill()
		return nil, err
	}

	outBytes, err := io.ReadAll(stdout)
	if err != nil {
		_ = cmd.Process.Kill()
		return nil, err
	}
	
	if se, _ := io.ReadAll(stderr); len(se) > 0 {
		log.Printf("cuda-worker stderr: %s", string(se))
	}

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("cuda process error: %w; out=%s", err, string(outBytes))
	}
	
	var resp map[string]interface{}
	if err := json.Unmarshal(outBytes, &resp); err != nil {
		return nil, fmt.Errorf("parse embed worker json: %w (raw=%s)", err, string(outBytes))
	}
	return resp, nil
}