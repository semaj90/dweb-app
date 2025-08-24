package fastjson

// Lightweight abstraction over JSON encoding to allow future drop-in replacement
// with a faster library (e.g. sonic, go-json) or a WASM bridge. Phase 1 keeps
// stdlib to minimize risk while centralizing call sites.

import (
	"bytes"
	"encoding/json"
	"os"
	"sync"
	"time"

	// Optional high-performance codecs
	"github.com/bytedance/sonic"
	gojson "github.com/goccy/go-json"
)

// Codec exposes the minimal surface we need. Additional methods can be added
// once alternative implementations are introduced.
type Codec interface {
    Marshal(v any) ([]byte, error)
    NewEncoder(buf *bytes.Buffer) *json.Encoder
    Name() string
}

type stdlibCodec struct{}

func (s stdlibCodec) Marshal(v any) ([]byte, error) { return json.Marshal(v) }
func (s stdlibCodec) NewEncoder(buf *bytes.Buffer) *json.Encoder { return json.NewEncoder(buf) }
func (s stdlibCodec) Name() string { return "encoding/json" }

var active Codec = stdlibCodec{}

// sonicCodec implements Codec using bytedance/sonic for Marshal (encoder falls back)
type sonicCodec struct{}
func (s sonicCodec) Marshal(v any) ([]byte, error) { return sonic.Marshal(v) }
// For streaming Encode we fall back to stdlib encoder to keep interface simplicity.
func (s sonicCodec) NewEncoder(buf *bytes.Buffer) *json.Encoder { return json.NewEncoder(buf) }
func (s sonicCodec) Name() string { return "sonic" }

// goJSONCodec uses goccy/go-json
type goJSONCodec struct{}
func (g goJSONCodec) Marshal(v any) ([]byte, error) { return gojson.Marshal(v) }
// We return a stdlib encoder for compatibility; Marshal benefits from go-json.
func (g goJSONCodec) NewEncoder(buf *bytes.Buffer) *json.Encoder { return json.NewEncoder(buf) }
func (g goJSONCodec) Name() string { return "go-json" }

// metrics (cheap, no locking on fast path except atomic increments if later added)
type Stats struct {
    Encodes        int64         `json:"encodes"`
    BytesProduced  int64         `json:"bytes_produced"`
    TotalEncodeDur time.Duration `json:"total_encode_duration_ns"`
    CodecName      string        `json:"codec"`
}

var stats Stats
var pool = sync.Pool{New: func() any { return new(bytes.Buffer) }}

// Marshal wraps active.Marshal and records lightweight timing & size metrics.
func Marshal(v any) ([]byte, error) {
    start := time.Now()
    b, err := active.Marshal(v)
    if err == nil {
        stats.Encodes++
        stats.BytesProduced += int64(len(b))
        stats.TotalEncodeDur += time.Since(start)
    }
    return b, err
}

// EncodeToBuffer obtains a pooled buffer, encodes v, and returns the buffer.
// Caller MUST call ReleaseBuffer when done. Buffer is not reset until release.
func EncodeToBuffer(v any) (*bytes.Buffer, error) {
    buf := pool.Get().(*bytes.Buffer)
    buf.Reset()
    enc := active.NewEncoder(buf)
    start := time.Now()
    if err := enc.Encode(v); err != nil {
        pool.Put(buf)
        return nil, err
    }
    stats.Encodes++
    stats.BytesProduced += int64(buf.Len())
    stats.TotalEncodeDur += time.Since(start)
    return buf, nil
}

// ReleaseBuffer returns buffer to pool.
func ReleaseBuffer(b *bytes.Buffer) { if b != nil { pool.Put(b) } }

// GetStats returns a snapshot of current stats.
func GetStats() Stats { stats.CodecName = active.Name(); return stats }

// Init allows selecting alternative codec via env var. For now only stdlib exists.
func Init() {
    // Placeholder: future HIGH_PERF_JSON=1 could load alternative implementation.
    switch os.Getenv("HIGH_PERF_JSON") {
    case "sonic":
        active = sonicCodec{}
    case "go-json", "gojson":
        active = goJSONCodec{}
    case "1": // temporary alias meaning "sonic" preferred
        active = sonicCodec{}
    default:
        // stdlib
    }
}

// SetCodec allows tests/benchmarks to switch codec programmatically.
// Accepted names: "stdlib", "sonic", "go-json".
func SetCodec(name string) {
    switch name {
    case "sonic":
        active = sonicCodec{}
    case "go-json", "gojson":
        active = goJSONCodec{}
    default:
        active = stdlibCodec{}
    }
}

func init() { Init() }
