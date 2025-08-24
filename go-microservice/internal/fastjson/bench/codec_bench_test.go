package bench

import (
	"encoding/json"
	fj "legal-ai-production/internal/fastjson"
	"testing"
)

type sample struct {
    ID    int      `json:"id"`
    Name  string   `json:"name"`
    Tags  []string `json:"tags"`
    Data  []int    `json:"data"`
}

var payload = sample{ID: 42, Name: "legal-ai", Tags: []string{"embedding","rag","stream"}, Data: func() []int { a := make([]int, 1024); for i := range a { a[i] = i } ; return a }() }

func BenchmarkStdlibMarshal(b *testing.B) {
    fj.SetCodec("stdlib")
    for i := 0; i < b.N; i++ {
        if _, err := fj.Marshal(payload); err != nil { b.Fatal(err) }
    }
}

func BenchmarkSonicMarshal(b *testing.B) {
    fj.SetCodec("sonic")
    for i := 0; i < b.N; i++ {
        if _, err := fj.Marshal(payload); err != nil { b.Fatal(err) }
    }
}

func BenchmarkGoJSONMarshal(b *testing.B) {
    fj.SetCodec("go-json")
    for i := 0; i < b.N; i++ {
        if _, err := fj.Marshal(payload); err != nil { b.Fatal(err) }
    }
}

func BenchmarkStdlibDirect(b *testing.B) {
    // control: bypass abstraction
    for i := 0; i < b.N; i++ {
        if _, err := json.Marshal(payload); err != nil { b.Fatal(err) }
    }
}
