package main

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	"google.golang.org/protobuf/proto"

	pb "legal-ai-production/proto/feedback"
)

var ctx = context.Background()

func float32SliceToBytes(f []float32) []byte {
	buf := make([]byte, 4*len(f))
	for i := range f {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(f[i]))
	}
	return buf
}

func floatSliceToPgvectorLiteral(f []float32) string {
	// simple literal builder: [0.1,0.2,...]
	if len(f) == 0 {
		return "[]"
	}
	out := "["
	for i, v := range f {
		if i != 0 {
			out += ","
		}
		out += fmt.Sprintf("%f", v)
	}
	out += "]"
	return out
}

func mapToJsonb(m map[string]string) []byte {
	if m == nil { return []byte("{}") }
	b, _ := json.Marshal(m)
	return b
}

func collectFeedback(db *sql.DB, rdb *redis.Client, userID, jobID, action string, vec []float32, meta map[string]string) error {
	id := uuid.New().String()
	ev := &pb.FeedbackEvent{
		Id: id,
		UserId: userID,
		JobId: jobID,
		TsUnixMs: time.Now().UnixMilli(),
		Action: action,
		Reward: 0.0,
		Vector: vec,
		Meta: meta,
	}

	data, err := proto.Marshal(ev)
	if err != nil { return err }

	vecStr := floatSliceToPgvectorLiteral(vec)
	metaJson := mapToJsonb(meta)

	_, err = db.ExecContext(ctx, `INSERT INTO feedback_events (id, user_id, job_id, ts, action, reward, meta, vec, proto)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
		id, userID, jobID, time.Now(), action, 0.0, metaJson, vecStr, data)
	if err != nil { return err }

	fields := map[string]interface{}{
		"id": id,
		"user_id": userID,
		"job_id": jobID,
		"proto": data,
	}
	if _, err := rdb.XAdd(ctx, &redis.XAddArgs{Stream: "feedback:events", Values: fields}).Result(); err != nil {
		return err
	}
	log.Printf("collected feedback %s\n", id)
	return nil
}

func main() {
	// Example main that connects to Postgres and Redis and sends a dummy feedback
	dbUrl := "postgresql://postgres:postgres@localhost:5432/legal_ai_db"
	db, err := sql.Open("pgx", dbUrl)
	if err != nil { log.Fatal(err) }
	defer db.Close()

	rdb := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	defer rdb.Close()

	vec := make([]float32, 768)
	for i := range vec { vec[i] = float32(i) * 0.001 }

	meta := map[string]string{"route":"search","path":"/cases/123"}

	if err := collectFeedback(db, rdb, "user-1", "job-1", "accept", vec, meta); err != nil {
		log.Fatal(err)
	}

	fmt.Println("sent sample feedback")
}
