package main

import (
	"hash/maphash"
	"testing"
)

// helper to create sample results
func sampleResults(n int) []RankingInput {
	res := make([]RankingInput,0,n)
	for i:=0;i<n;i++ {
		res = append(res, RankingInput{DocID: uint64(i+1), Score: float32(0.9 - float32(i)*0.01), Flags: uint8(i%4), Summary: "summary text", URL: "http://x"})
	}
	return res
}

func TestPackDecodeRoundTrip(t *testing.T) {
	rc := NewRankingCache()
	inputs := sampleResults(10)
	h := rc.hashInputs(inputs)
	blob, err := rc.packRankings(inputs, h)
	if err != nil { t.Fatalf("pack error: %v", err) }
	v, ch, decoded, err := decodePacked(blob)
	if err != nil { t.Fatalf("decode error: %v", err) }
	if v != rankingPackVersion { t.Errorf("version mismatch: got %d", v) }
	if ch != h { t.Errorf("hash mismatch") }
	if len(decoded) != len(inputs) { t.Fatalf("count mismatch decoded=%d", len(decoded)) }
	for i,d := range decoded { if d.DocID != inputs[i].DocID { t.Fatalf("doc id mismatch at %d", i) } }
}

func TestCRCDetection(t *testing.T) {
	rc := NewRankingCache()
	inputs := sampleResults(5)
	h := rc.hashInputs(inputs)
	blob, err := rc.packRankings(inputs, h)
	if err != nil { t.Fatalf("pack error: %v", err) }
	// corrupt last byte
	blob[len(blob)-1] ^= 0xFF
	_,_,_, err = decodePacked(blob)
	if err == nil { t.Fatalf("expected CRC error but got nil") }
}

func TestAutoPublishFromSearch(t *testing.T) {
	rc := NewRankingCache()
	results := []map[string]interface{}{
		{"document_id":"1","content":"Alpha content","score":0.91},
		{"document_id":"2","content":"Beta content","score":0.73},
	}
	key, meta, err := rc.AutoPublishFromSearch(results)
	if err != nil { t.Fatalf("autopublish error: %v", err) }
	if key == "" { t.Fatalf("empty key") }
	if meta.Count != 2 { t.Fatalf("count mismatch: %d", meta.Count) }
}

func TestHashStability(t *testing.T) {
	rc := NewRankingCache()
	inputs := sampleResults(6)
	h1 := rc.hashInputs(inputs)
	// reorder intentionally (hashInputs sorts internally)
	inputs[0], inputs[5] = inputs[5], inputs[0]
	h2 := rc.hashInputs(inputs)
	if h1 != h2 { t.Fatalf("hash not stable across order changes") }
}

func TestSummaryAndURLRegistry(t *testing.T) {
	rc := NewRankingCache()
	inputs := []RankingInput{{DocID:1, Score:0.8, Flags:1, Summary:"sumA", URL:"http://a"}}
	h := rc.hashInputs(inputs)
	_, err := rc.packRankings(inputs, h)
	if err != nil { t.Fatalf("pack err: %v", err) }
	if len(rc.summaries)==0 || len(rc.urls)==0 { t.Fatalf("registries not populated") }
}

func BenchmarkPackRankings(b *testing.B) {
	rc := NewRankingCache()
	inputs := sampleResults(128)
	for i:=0;i<b.N;i++ {
		_ , _ = rc.packRankings(inputs, uint64(i))
	}
}

func BenchmarkHashInputs(b *testing.B) {
	rc := NewRankingCache()
	inputs := sampleResults(256)
	for i:=0;i<b.N;i++ { _ = rc.hashInputs(inputs) }
}

// quick check that maphash seed changes still allow hashInputs to function
func TestMaphashSeedUsage(t *testing.T) {
	rc := NewRankingCache()
	inputs := sampleResults(4)
	h1 := rc.hashInputs(inputs)
	// new seed
	rc.maphashSeed = maphash.MakeSeed()
	h2 := rc.hashInputs(inputs)
	if h1 == h2 { t.Logf("(info) identical hash after seed change - acceptable but note") }
}
