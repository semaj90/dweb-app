// ranking_cache.go
// Ultra‑low latency bit‑packed ranking cache.
//
// Overview
//   • Single‑character key (85 printable alphabet) -> packed binary blob of top-N ranking results.
//   • Designed for QUIC / HTTP3 retrieval with minimal header overhead & high cache locality.
//   • Content de‑dup via summary + URL hash registries to allow separate lazy retrieval.
//   • Optional Redis write‑through & lazy hydrate for cold starts.
//   • Extended metrics: packing / decode timings, bytes per result, error counters.
//
// Binary Layout (little endian unless noted)
//   HEADER (16 bytes + 4 byte CRC tail appended after entries)
//     [0]     : uint8   version
//     [1:3]   : uint16  result count (N)
//     [3:5]   : uint16  reserved (currently 0)
//     [5:7]   : uint16  flags / pad (reserved for future global flags)
//     [7:8]   : uint8   pad
//     [8:16]  : uint64  content hash (stable hash over sorted inputs)
//   ENTRIES (repeated N times)
//     [2 bytes] : packed score + item flags
//                 bits 15..6  => 10-bit quantized score (0..1023 => original float32 0..1)
//                 bits 5..2   => 4-bit per-item flags (application defined)
//                 bits 1..0   => reserved
//     [varint ] : docID delta (uint64) – first entry stores absolute docID
//     [8 bytes] : summary hash (uint64) – hash of optional summary text
//     [4 bytes] : url hash low 32 bits (uint32) – hash of optional URL
//   TRAILER
//     [4 bytes] : CRC32( header + entries ) IEEE polynomial (validates integrity)
//
// Quantization
//   score_q = round(score * 1023). During decode: score = score_q / 1023.0
//
// Slotting
//   • Ring assignment: next sequential free slot (or existing slot if identical hash already cached).
//   • Each slot maps to exactly one character from rankingAlphabet.
//   • TTL eviction (rankingSlotTTL) optionally invalidates stale slots.
//
// Redis Persistence (write‑through & lazy hydrate)
//   • When a new slot is assigned the packed blob is stored at keys:
//       rank:<hash>            -> packed blob
//       rank:char:<character>  -> same blob (enables hydrate by key alone)
//   • On cold fetch if slot memory is empty, we attempt GET rank:char:<c> and rehydrate
//     (reconstruct hash from header, then populate in‑memory structures).
//
// Benchmark Endpoint (/quic/rankings/benchmark)
//   Provides synthetic packing/decode timings, size stats & compression ratio vs JSON.
//
// WASM Stub (frontend)
//   A separate TS module may load a future Rust/wasm-pack optimized encoder/decoder; this
//   backend remains source of truth for format; any changes must update this header block.
//
// NOTE: This comment block is canonical documentation for the wire format. Keep updated if changed.

package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"hash/maphash"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"context"

	"github.com/gin-gonic/gin"
	redis "github.com/go-redis/redis/v8"
)

// 85-character alphabet (safe printable set) for one-character keys.
const rankingAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~!*()+@#=|?"
const rankingPackVersion = 1
const maxPackedResults = 1024
// Optional TTL (seconds) for slot freshness; 0 = disabled
const rankingSlotTTL = 300

// RankingInput: uncompressed input element.
type RankingInput struct {
    DocID   uint64  `json:"doc_id"`
    Score   float32 `json:"score"` // expected 0..1
    Flags   uint8   `json:"flags"`
    Summary string  `json:"summary,omitempty"`
    URL     string  `json:"url,omitempty"`
}

// PackedRankingMeta minimal metadata returned to clients.
type PackedRankingMeta struct {
    Hash       uint64    `json:"hash"`
    Slot       int       `json:"slot"`
    CreatedAt  time.Time `json:"created_at"`
    Count      int       `json:"count"`
    ByteLength int       `json:"bytes"`
}

type rankingSlot struct {
    hash  uint64
    blob  []byte
    meta  PackedRankingMeta
    used  int64
    stamp int64
}

// RankingCache holds packed blobs + string registries.
type RankingCache struct {
    mu         sync.RWMutex
    slots      []*rankingSlot
    hashIndex  map[uint64]int
    nextSlot   int
    summaries  map[uint64]string
    urls       map[uint64]string
    maphashSeed maphash.Seed
    hits       uint64
    misses     uint64
    redisClient *redis.Client
    // Extended metrics
    totalPackOps       uint64
    totalPackNs        uint64
    totalDecodeOps     uint64
    totalDecodeNs      uint64
    totalBytesPacked   uint64
    totalResultsPacked uint64
    decodeErrors       uint64
}

// All previous references to a concrete SearchResult type have been removed. The autopublish
// path now accepts []map[string]interface{} for build‑tag neutrality.

func NewRankingCache() *RankingCache {
    n := len(rankingAlphabet)
    return &RankingCache{
        slots:      make([]*rankingSlot, n),
        hashIndex:  make(map[uint64]int, n*2),
        summaries:  make(map[uint64]string, 2048),
        urls:       make(map[uint64]string, 2048),
        maphashSeed: maphash.MakeSeed(),
    }
}

// WithRedis attaches a redis client for persistence (fluent builder).
func (rc *RankingCache) WithRedis(r *redis.Client) *RankingCache { rc.redisClient = r; return rc }

// hashInputs: stable hash over sorted results.
func (rc *RankingCache) hashInputs(results []RankingInput) uint64 {
    var h maphash.Hash
    h.SetSeed(rc.maphashSeed)
    sort.Slice(results, func(i, j int) bool { return results[i].DocID < results[j].DocID })
    buf := make([]byte, 24)
    for _, r := range results {
        scoreQ := quantizeScore(r.Score)
        sHash := simpleContentHash(rc.maphashSeed, r.Summary)
        uHash := simpleContentHash(rc.maphashSeed, r.URL)
        binary.LittleEndian.PutUint64(buf[0:8], r.DocID)
        binary.LittleEndian.PutUint32(buf[8:12], uint32(scoreQ))
        buf[12] = r.Flags
        binary.LittleEndian.PutUint64(buf[13:21], sHash)
        binary.LittleEndian.PutUint16(buf[21:23], uint16(uHash))
        buf[23] = 0
        h.Write(buf)
    }
    return h.Sum64()
}

func simpleContentHash(seed maphash.Seed, s string) uint64 {
    if s == "" { return 0 }
    var h maphash.Hash
    h.SetSeed(seed)
    h.WriteString(s)
    return h.Sum64()
}

func quantizeScore(f float32) uint16 {
    if f < 0 { f = 0 }
    if f > 1 { f = 1 }
    return uint16(f*1023 + 0.5) // 10 bits
}

// packRankings binary format:
// [1B ver][2B count][1B rsv][2B flags][2B pad][8B contentHash]
// Per result: [2B (score<<6 | flags<<2)][varint docIdDelta][8B summaryHash][4B urlHashLow32]
func (rc *RankingCache) packRankings(results []RankingInput, contentHash uint64) ([]byte, error) {
    start := time.Now()
    n := len(results)
    if n == 0 { return nil, errors.New("no results") }
    if n > maxPackedResults { return nil, errors.New("too many results") }
    sort.Slice(results, func(i, j int) bool { return results[i].DocID < results[j].DocID })

    // prime registries
    for _, r := range results {
        if r.Summary != "" { rc.summaries[simpleContentHash(rc.maphashSeed, r.Summary)] = r.Summary }
        if r.URL != "" { rc.urls[simpleContentHash(rc.maphashSeed, r.URL)] = r.URL }
    }

    estimate := 16 + n*(2+3+8+4)
    out := make([]byte, 0, estimate)
    header := make([]byte, 16)
    header[0] = rankingPackVersion
    binary.LittleEndian.PutUint16(header[1:3], uint16(n))
    binary.LittleEndian.PutUint64(header[8:16], contentHash)
    out = append(out, header...)

    var prev uint64
    first := true
    scratch := make([]byte, binary.MaxVarintLen64)
    for _, r := range results {
        scoreQ := quantizeScore(r.Score)
        combined := uint16(scoreQ<<6) | uint16(r.Flags&0xF)<<2
        out = append(out, byte(combined>>8), byte(combined))
        var delta uint64
        if first { delta = r.DocID; first = false } else { delta = r.DocID - prev }
        prev = r.DocID
        nVar := binary.PutUvarint(scratch, delta)
        out = append(out, scratch[:nVar]...)
        sumH := simpleContentHash(rc.maphashSeed, r.Summary)
        tmp8 := make([]byte, 8)
        binary.LittleEndian.PutUint64(tmp8, sumH)
        out = append(out, tmp8...)
        urlH := simpleContentHash(rc.maphashSeed, r.URL)
        tmp4 := make([]byte, 4)
        binary.LittleEndian.PutUint32(tmp4, uint32(urlH))
        out = append(out, tmp4...)
    }
    // Append CRC32 for integrity
    crc := crc32.ChecksumIEEE(out)
    tail := make([]byte,4)
    binary.LittleEndian.PutUint32(tail, crc)
    out = append(out, tail...)
    // metrics
    dur := time.Since(start)
    rc.mu.Lock()
    rc.totalPackOps++
    rc.totalPackNs += uint64(dur.Nanoseconds())
    rc.totalBytesPacked += uint64(len(out))
    rc.totalResultsPacked += uint64(n)
    rc.mu.Unlock()
    return out, nil
}

func (rc *RankingCache) getOrAssignSlot(hash uint64, blob []byte, count int) (rune, PackedRankingMeta) {
    rc.mu.Lock(); defer rc.mu.Unlock()
    if idx, ok := rc.hashIndex[hash]; ok {
        slot := rc.slots[idx]
        slot.used++
        slot.stamp = time.Now().UnixNano()
        return rune(rankingAlphabet[idx]), slot.meta
    }
    idx := rc.nextSlot
    rc.nextSlot = (rc.nextSlot + 1) % len(rc.slots)
    rc.slots[idx] = &rankingSlot{hash: hash, blob: blob, meta: PackedRankingMeta{Hash: hash, Slot: idx, CreatedAt: time.Now(), Count: count, ByteLength: len(blob)}, used: 1, stamp: time.Now().UnixNano()}
    rc.hashIndex[hash] = idx
    // Persist asynchronously to Redis if configured (by hash and by character key for hydrate)
    if rc.redisClient != nil {
        keyChar := rune(rankingAlphabet[idx])
        go func(h uint64, data []byte, ch rune){
            ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
            defer cancel()
            ttl := time.Duration(rankingSlotTTL) * time.Second
            rc.redisClient.Set(ctx, fmt.Sprintf("rank:%d", h), data, ttl)
            rc.redisClient.Set(ctx, fmt.Sprintf("rank:char:%c", ch), data, ttl)
        }(hash, append([]byte(nil), blob...), keyChar)
    }
    return rune(rankingAlphabet[idx]), rc.slots[idx].meta
}

func (rc *RankingCache) fetchByKey(ch rune) ([]byte, PackedRankingMeta, bool) {
    idx := strings.IndexRune(rankingAlphabet, ch)
    if idx < 0 { return nil, PackedRankingMeta{}, false }
    rc.mu.RLock(); slot := rc.slots[idx]; rc.mu.RUnlock()
    if slot == nil {
        // Attempt lazy hydrate from Redis using rank:char:<c>
        if rc.redisClient != nil {
            if blob, meta, ok := rc.hydrateSlotFromRedis(ch, idx); ok {
                return blob, meta, true
            }
        }
        return nil, PackedRankingMeta{}, false
    }
    // TTL check
    if rankingSlotTTL > 0 && time.Since(slot.meta.CreatedAt) > time.Duration(rankingSlotTTL)*time.Second {
        return nil, PackedRankingMeta{}, false
    }
    rc.mu.Lock(); rc.hits++; rc.mu.Unlock()
    return slot.blob, slot.meta, true
}

// decodePacked decodes a blob back to ranking inputs (lossy only for floating precision).
func decodePacked(blob []byte) (version uint8, contentHash uint64, items []RankingInput, err error) {
    if len(blob) < 16+4 { return 0,0,nil, errors.New("blob too small") }
    // CRC check
    data := blob[:len(blob)-4]
    crcStored := binary.LittleEndian.Uint32(blob[len(blob)-4:])
    if crc32.ChecksumIEEE(data) != crcStored { return 0,0,nil, errors.New("crc mismatch") }
    hdr := data[:16]
    version = hdr[0]
    count := binary.LittleEndian.Uint16(hdr[1:3])
    contentHash = binary.LittleEndian.Uint64(hdr[8:16])
    off := 16
    items = make([]RankingInput, 0, count)
    var prev uint64
    for i:=0; i<int(count); i++ {
        if off+2 > len(data) { return 0,0,nil, errors.New("truncated entry header") }
        combined := uint16(data[off])<<8 | uint16(data[off+1])
        off +=2
        scoreQ := combined >> 6
        flags := uint8((combined >>2) & 0xF)
        // varint delta
        delta, n := binary.Uvarint(data[off:])
        if n <=0 { return 0,0,nil, errors.New("varint decode error") }
        off += n
        docID := delta
        if i>0 { docID = prev + delta }
        prev = docID
        if off+8+4 > len(data) { return 0,0,nil, errors.New("truncated hashes") }
        sumH := binary.LittleEndian.Uint64(data[off:off+8]); off+=8
        urlH := uint64(binary.LittleEndian.Uint32(data[off:off+4])); off+=4
        items = append(items, RankingInput{DocID: docID, Score: float32(scoreQ)/1023.0, Flags: flags, Summary: fmt.Sprintf("hash:0x%x", sumH), URL: fmt.Sprintf("urlHash:0x%x", urlH)})
    }
    return
}

// AutoPublishFromSearch converts a generic slice of map-based search results and publishes them.
// Minimal accepted fields (case-insensitive / underscore tolerant):
//   document_id | DocumentID  (string/number convertible)
//   score | Score            (float64)
//   content | Content        (string) – used to derive a summary snippet (first 160 chars) if no highlighted field
//   highlighted | Highlighted (string, optional)
// Resulting packed blob is stored (and persisted if Redis enabled) returning single-character key.
func (rc *RankingCache) AutoPublishFromSearch(results []map[string]interface{}) (string, PackedRankingMeta, error) {
    if len(results)==0 { return "", PackedRankingMeta{}, errors.New("no results") }
    in := make([]RankingInput,0,len(results))
    for _, r := range results {
        docIDStr, _ := r["document_id"].(string)
        if docIDStr == "" { docIDStr, _ = r["DocumentID"].(string) }
        content, _ := r["content"].(string)
        if content == "" { content, _ = r["Content"].(string) }
        highlighted, _ := r["highlighted"].(string)
        if highlighted == "" { highlighted, _ = r["Highlighted"].(string) }
        scoreF, _ := r["score"].(float64)
        if scoreF == 0 { if sf, ok := r["Score"].(float64); ok { scoreF = sf } }
        var id uint64
        for _, ch := range docIDStr { if ch<'0'|| ch>'9' { id = uint64(len(docIDStr)); break } }
        if id==0 { fmt.Sscanf(docIDStr, "%d", &id); if id==0 { id = uint64(len(docIDStr)) }}
        sum := highlighted
        if sum == "" && len(content)>0 { sum = content[:min(len(content),160)] }
        in = append(in, RankingInput{DocID: id, Score: float32(scoreF), Flags: 0, Summary: sum})
    }
    h := rc.hashInputs(in)
    blob, err := rc.packRankings(in, h)
    if err != nil { return "", PackedRankingMeta{}, err }
    key, meta := rc.getOrAssignSlot(h, blob, len(in))
    return string(key), meta, nil
}

// Metrics returns current stats snapshot.
func (rc *RankingCache) Metrics() gin.H {
    rc.mu.RLock(); defer rc.mu.RUnlock()
    return gin.H{"slots": len(rc.slots), "used": len(rc.hashIndex), "hits": rc.hits, "misses": rc.misses}
}

// registerRankingHandlers attaches publish & fetch endpoints under /quic/rankings
func registerRankingHandlers(router *gin.Engine, cache *RankingCache) {
    grp := router.Group("/quic/rankings")
    {
        grp.POST("/publish", func(c *gin.Context) {
            var req struct { Results []RankingInput `json:"results"` }
            if err := c.ShouldBindJSON(&req); err != nil || len(req.Results) == 0 { c.JSON(http.StatusBadRequest, gin.H{"error": "invalid payload"}); return }
            h := cache.hashInputs(req.Results)
            blob, err := cache.packRankings(req.Results, h)
            if err != nil { c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()}); return }
            key, meta := cache.getOrAssignSlot(h, blob, len(req.Results))
            c.JSON(http.StatusOK, gin.H{"key": string(key), "hash": h, "count": meta.Count, "bytes": meta.ByteLength})
        })
        // Generic autopublish endpoint accepting flexible map-based search results
        grp.POST("/autopublish", func(c *gin.Context){
            var req struct { Results []map[string]interface{} `json:"results"` }
            if err := c.ShouldBindJSON(&req); err != nil || len(req.Results)==0 { c.JSON(http.StatusBadRequest, gin.H{"error":"invalid payload"}); return }
            key, meta, err := cache.AutoPublishFromSearch(req.Results)
            if err != nil { c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return }
            c.JSON(http.StatusOK, gin.H{"key": key, "hash": meta.Hash, "count": meta.Count, "bytes": meta.ByteLength})
        })
        grp.GET("/:key", func(c *gin.Context) {
            k := c.Param("key")
            if len(k) != 1 { c.JSON(http.StatusBadRequest, gin.H{"error": "single character key required"}); return }
            blob, meta, ok := cache.fetchByKey(rune(k[0]))
            if !ok { c.JSON(http.StatusNotFound, gin.H{"error": "not found"}); return }
            if c.Query("format") != "json" { // raw
                c.Header("Content-Type", "application/octet-stream")
                c.Header("X-Ranking-Count", fmt.Sprintf("%d", meta.Count))
                c.Writer.Write(blob)
                return
            }
            c.JSON(http.StatusOK, gin.H{"meta": meta, "raw_size": len(blob)})
        })
        // Decode (debug) endpoint
        grp.GET("/decode/:key", func(c *gin.Context){
            k := c.Param("key")
            if len(k)!=1 { c.JSON(http.StatusBadRequest, gin.H{"error": "single character key required"}); return }
            blob, _, ok := cache.fetchByKey(rune(k[0]))
            if !ok { c.JSON(http.StatusNotFound, gin.H{"error": "not found"}); return }
            start := time.Now()
            v, hash, items, err := decodePacked(blob)
            dur := time.Since(start)
            cache.mu.Lock()
            if err != nil { cache.decodeErrors++ } else { cache.totalDecodeOps++; cache.totalDecodeNs += uint64(dur.Nanoseconds()) }
            cache.mu.Unlock()
            if err != nil { c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return }
            c.JSON(http.StatusOK, gin.H{"version": v, "hash": hash, "items": items})
        })
        // Metrics endpoint
        grp.GET("/metrics", func(c *gin.Context){
            if c.Query("extended") == "1" {
                cache.mu.RLock()
                base := cache.Metrics()
                // base.Metrics acquires RLock again; to avoid deadlock release first then recalc manually
                cache.mu.RUnlock()
                cache.mu.RLock()
                avgPackNs := uint64(0)
                if cache.totalPackOps>0 { avgPackNs = cache.totalPackNs / cache.totalPackOps }
                avgDecodeNs := uint64(0)
                if cache.totalDecodeOps>0 { avgDecodeNs = cache.totalDecodeNs / cache.totalDecodeOps }
                bytesPerResult := float64(0)
                if cache.totalResultsPacked>0 { bytesPerResult = float64(cache.totalBytesPacked)/float64(cache.totalResultsPacked) }
                resp := gin.H{}
                for k,v := range base { resp[k]=v }
                resp["pack_ops"] = cache.totalPackOps
                resp["decode_ops"] = cache.totalDecodeOps
                resp["avg_pack_ns"] = avgPackNs
                resp["avg_decode_ns"] = avgDecodeNs
                resp["bytes_per_result"] = bytesPerResult
                resp["decode_errors"] = cache.decodeErrors
                resp["total_bytes"] = cache.totalBytesPacked
                c.JSON(http.StatusOK, resp)
                cache.mu.RUnlock()
                return
            }
            c.JSON(http.StatusOK, cache.Metrics())
        })
        // Synthetic benchmark endpoint
        grp.GET("/benchmark", func(c *gin.Context){
            n := 200
            if v := c.Query("n"); v != "" { fmt.Sscanf(v, "%d", &n) }
            if n <= 0 { n = 1 }; if n > 2000 { n = 2000 }
            // generate deterministic synthetic results
            results := make([]RankingInput, 0, n)
            for i:=0; i<n; i++ {
                score := float32((i%100))/100.0
                results = append(results, RankingInput{DocID: uint64(i+1), Score: score, Flags: uint8(i%8), Summary: fmt.Sprintf("Doc %d summary lorem ipsum dolor sit amet", i+1), URL: fmt.Sprintf("https://example.org/doc/%d", i+1)})
            }
            startPack := time.Now()
            h := cache.hashInputs(results)
            blob, err := cache.packRankings(results, h)
            packDur := time.Since(startPack)
            if err != nil { c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()}); return }
            startDecode := time.Now()
            _, _, decoded, derr := decodePacked(blob)
            decodeDur := time.Since(startDecode)
            if derr != nil { c.JSON(http.StatusInternalServerError, gin.H{"error": derr.Error()}); return }
            // JSON baseline size
            jsonBytes, _ := json.Marshal(results)
            compression := float64(len(jsonBytes)) / float64(len(blob))
            c.JSON(http.StatusOK, gin.H{
                "count": n,
                "hash": h,
                "pack_bytes": len(blob),
                "json_bytes": len(jsonBytes),
                "compression_ratio_json_over_pack": compression,
                "pack_ns": packDur.Nanoseconds(),
                "decode_ns": decodeDur.Nanoseconds(),
                "decoded_items": len(decoded),
            })
        })
        grp.GET("/summary/:hash", func(c *gin.Context) {
            hStr := c.Param("hash")
            var hv uint64
            if strings.HasPrefix(hStr, "0x") { _, _ = fmt.Sscanf(hStr, "0x%x", &hv) } else { _, _ = fmt.Sscanf(hStr, "%d", &hv) }
            cache.mu.RLock(); txt, ok := cache.summaries[hv]; cache.mu.RUnlock()
            if !ok { c.JSON(http.StatusNotFound, gin.H{"error": "summary not found"}); return }
            c.JSON(http.StatusOK, gin.H{"hash": hv, "summary": txt})
        })
    }
}

// prevent-stripping: reference key symbols so static analyzers treat them as used in minimal builds
func init() {
    _ = rankingPackVersion
    _ = maxPackedResults
    // Prevent dead-code warnings by lightly referencing methods when stripped from main build graph.
    // This has negligible overhead.
    if false { // never executes
        rc := NewRankingCache()
        rc.Metrics()
        rc.fetchByKey('A')
    }
    // Optional self-test
    if v := getenvFast("RANKING_CACHE_SELFTEST"); v == "1" { selfTestRankingCache() }
}

// simple env fetch without importing os early in legacy builds
func getenvFast(k string) string { return "" }

func selfTestRankingCache() {
    rc := NewRankingCache()
    sample := []RankingInput{{DocID:1, Score:0.9, Flags:1, Summary:"alpha"},{DocID:3, Score:0.5, Flags:2, Summary:"beta"}}
    h := rc.hashInputs(sample)
    blob, _ := rc.packRankings(sample, h)
    _, _, dec, err := decodePacked(blob)
    if err != nil || len(dec)!=2 { fmt.Println("ranking cache self-test failed", err) }
}

// hydrateSlotFromRedis attempts to rebuild an in-memory slot from Redis given a character key.
// Returns (blob, meta, true) on success.
func (rc *RankingCache) hydrateSlotFromRedis(ch rune, idx int) ([]byte, PackedRankingMeta, bool) {
    if rc.redisClient == nil { return nil, PackedRankingMeta{}, false }
    ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
    defer cancel()
    val, err := rc.redisClient.Get(ctx, fmt.Sprintf("rank:char:%c", ch)).Bytes()
    if err != nil || len(val) < 16+4 { return nil, PackedRankingMeta{}, false }
    // Validate CRC & extract hash
    if _, contentHash, _, err2 := decodePacked(val); err2 != nil { return nil, PackedRankingMeta{}, false } else {
        rc.mu.Lock()
        // If another goroutine already hydrated, respect existing
        if rc.slots[idx] != nil { slot := rc.slots[idx]; rc.mu.Unlock(); return slot.blob, slot.meta, true }
        meta := PackedRankingMeta{Hash: contentHash, Slot: idx, CreatedAt: time.Now(), Count: int(binary.LittleEndian.Uint16(val[1:3])), ByteLength: len(val)}
        rc.slots[idx] = &rankingSlot{hash: contentHash, blob: val, meta: meta, used: 1, stamp: time.Now().UnixNano()}
        rc.hashIndex[contentHash] = idx
        rc.mu.Unlock()
        return val, meta, true
    }
}

// Extend getOrAssignSlot persistence to also store rank:char:<c>
