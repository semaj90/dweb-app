// ranking_cache_decoder.go
// Decoder (reverse expand) functionality for debugging and cache retrieval
// Unpacks bit-packed ranking cache blobs back to readable ranking results

package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"math"
)

// DecodedRanking represents an unpacked ranking result
type DecodedRanking struct {
	DocID   uint64  `json:"doc_id"`
	Score   float32 `json:"score"`
	Flags   uint8   `json:"flags"`
	Summary string  `json:"summary,omitempty"`
	URL     string  `json:"url,omitempty"`
}

// DecodedRankingResults contains unpacked results with metadata
type DecodedRankingResults struct {
	Version     uint8            `json:"version"`
	Count       int              `json:"count"`
	ContentHash uint64           `json:"content_hash"`
	Results     []DecodedRanking `json:"results"`
	BytesUsed   int              `json:"bytes_used"`
}

// unpackRankings decodes bit-packed ranking cache blobs
func (rc *RankingCache) unpackRankings(blob []byte) (*DecodedRankingResults, error) {
	if len(blob) < 16 {
		return nil, errors.New("blob too small for header")
	}

	// Parse header: [1B ver][2B count][1B rsv][2B flags][2B pad][8B contentHash]
	version := blob[0]
	if version != rankingPackVersion {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	count := binary.LittleEndian.Uint16(blob[1:3])
	contentHash := binary.LittleEndian.Uint64(blob[8:16])

	if count == 0 {
		return &DecodedRankingResults{
			Version:     version,
			Count:       0,
			ContentHash: contentHash,
			Results:     []DecodedRanking{},
			BytesUsed:   16,
		}, nil
	}

	// Decode results
	results := make([]DecodedRanking, 0, count)
	offset := 16
	var prevDocID uint64

	for i := 0; i < int(count); i++ {
		if offset+2 >= len(blob) {
			return nil, fmt.Errorf("unexpected end of blob at result %d", i)
		}

		// Parse score and flags: [2B (score<<6 | flags<<2)]
		combined := binary.LittleEndian.Uint16(blob[offset : offset+2])
		scoreQ := (combined >> 6) & 0x3FF  // 10 bits for score
		flags := (combined >> 2) & 0xF     // 4 bits for flags
		offset += 2

		// Dequantize score
		score := float32(scoreQ) / 1023.0

		// Parse docID delta (varint)
		delta, bytesRead := binary.Uvarint(blob[offset:])
		if bytesRead <= 0 {
			return nil, fmt.Errorf("failed to read docID varint at result %d", i)
		}
		offset += bytesRead

		var docID uint64
		if i == 0 {
			docID = delta
		} else {
			docID = prevDocID + delta
		}
		prevDocID = docID

		// Parse summary hash (8 bytes)
		if offset+8 > len(blob) {
			return nil, fmt.Errorf("unexpected end of blob reading summary hash at result %d", i)
		}
		summaryHash := binary.LittleEndian.Uint64(blob[offset : offset+8])
		offset += 8

		// Parse URL hash (4 bytes)
		if offset+4 > len(blob) {
			return nil, fmt.Errorf("unexpected end of blob reading url hash at result %d", i)
		}
		urlHashLow32 := binary.LittleEndian.Uint32(blob[offset : offset+4])
		urlHash := uint64(urlHashLow32) // Reconstruct from low 32 bits
		offset += 4

		// Lookup summary and URL from registries
		rc.mu.RLock()
		summary := rc.summaries[summaryHash]
		url := rc.urls[urlHash]
		rc.mu.RUnlock()

		results = append(results, DecodedRanking{
			DocID:   docID,
			Score:   score,
			Flags:   uint8(flags),
			Summary: summary,
			URL:     url,
		})
	}

	return &DecodedRankingResults{
		Version:     version,
		Count:       len(results),
		ContentHash: contentHash,
		Results:     results,
		BytesUsed:   offset,
	}, nil
}

// dequantizeScore converts 10-bit quantized score back to float32
func dequantizeScore(quantized uint16) float32 {
	if quantized > 1023 {
		quantized = 1023
	}
	return float32(quantized) / 1023.0
}

// debugRankingBlob provides detailed debugging information about a packed blob
func (rc *RankingCache) debugRankingBlob(blob []byte) string {
	if len(blob) < 16 {
		return "ERROR: Blob too small for header"
	}

	version := blob[0]
	count := binary.LittleEndian.Uint16(blob[1:3])
	reserved := blob[3]
	flags := binary.LittleEndian.Uint16(blob[4:6])
	pad := binary.LittleEndian.Uint16(blob[6:8])
	contentHash := binary.LittleEndian.Uint64(blob[8:16])

	debug := fmt.Sprintf("=== RANKING BLOB DEBUG ===\n")
	debug += fmt.Sprintf("Total Size: %d bytes\n", len(blob))
	debug += fmt.Sprintf("Header (16 bytes):\n")
	debug += fmt.Sprintf("  Version: %d\n", version)
	debug += fmt.Sprintf("  Count: %d\n", count)
	debug += fmt.Sprintf("  Reserved: %d\n", reserved)
	debug += fmt.Sprintf("  Flags: %d (0x%04X)\n", flags, flags)
	debug += fmt.Sprintf("  Padding: %d\n", pad)
	debug += fmt.Sprintf("  Content Hash: %d (0x%016X)\n", contentHash, contentHash)
	debug += fmt.Sprintf("\nPayload: %d bytes\n", len(blob)-16)

	if count == 0 {
		debug += "No results to decode.\n"
		return debug
	}

	// Attempt to decode first few results for debugging
	offset := 16
	debug += fmt.Sprintf("\nFirst few results:\n")

	for i := 0; i < int(math.Min(float64(count), 3)); i++ {
		if offset+14 >= len(blob) { // Minimum size for one result
			debug += fmt.Sprintf("  Result %d: ERROR - insufficient data\n", i)
			break
		}

		// Parse combined score/flags
		combined := binary.LittleEndian.Uint16(blob[offset : offset+2])
		scoreQ := (combined >> 6) & 0x3FF
		flags := (combined >> 2) & 0xF
		score := dequantizeScore(scoreQ)

		debug += fmt.Sprintf("  Result %d:\n", i)
		debug += fmt.Sprintf("    Combined: 0x%04X\n", combined)
		debug += fmt.Sprintf("    Score: %.3f (quantized: %d)\n", score, scoreQ)
		debug += fmt.Sprintf("    Flags: 0x%X\n", flags)

		offset += 2

		// Show varint docID delta bytes
		varintStart := offset
		_, bytesRead := binary.Uvarint(blob[offset:])
		if bytesRead <= 0 {
			debug += fmt.Sprintf("    DocID Delta: ERROR - invalid varint\n")
			break
		}
		debug += fmt.Sprintf("    DocID Delta Varint: %d bytes\n", bytesRead)
		offset += bytesRead

		// Show hash bytes
		if offset+12 <= len(blob) {
			summaryHash := binary.LittleEndian.Uint64(blob[offset : offset+8])
			urlHash := binary.LittleEndian.Uint32(blob[offset+8 : offset+12])
			debug += fmt.Sprintf("    Summary Hash: 0x%016X\n", summaryHash)
			debug += fmt.Sprintf("    URL Hash: 0x%08X\n", urlHash)
			offset += 12
		} else {
			debug += fmt.Sprintf("    Hashes: ERROR - insufficient data\n")
			break
		}

		debug += fmt.Sprintf("    Total bytes for result %d: %d\n", i, offset-varintStart+2)
	}

	if count > 3 {
		debug += fmt.Sprintf("... and %d more results\n", count-3)
	}

	return debug
}

// validateRankingBlob performs integrity checks on a packed blob
func (rc *RankingCache) validateRankingBlob(blob []byte) []string {
	var issues []string

	if len(blob) < 16 {
		issues = append(issues, "Blob size too small for header")
		return issues
	}

	version := blob[0]
	if version != rankingPackVersion {
		issues = append(issues, fmt.Sprintf("Unsupported version: %d (expected: %d)", version, rankingPackVersion))
	}

	count := binary.LittleEndian.Uint16(blob[1:3])
	if count > maxPackedResults {
		issues = append(issues, fmt.Sprintf("Count exceeds maximum: %d (max: %d)", count, maxPackedResults))
	}

	// Try to decode and check for structural issues
	decoded, err := rc.unpackRankings(blob)
	if err != nil {
		issues = append(issues, fmt.Sprintf("Decoding error: %v", err))
		return issues
	}

	if len(decoded.Results) != int(count) {
		issues = append(issues, fmt.Sprintf("Result count mismatch: decoded %d, header says %d", 
			len(decoded.Results), count))
	}

	// Check for sorted docIDs
	for i := 1; i < len(decoded.Results); i++ {
		if decoded.Results[i].DocID <= decoded.Results[i-1].DocID {
			issues = append(issues, fmt.Sprintf("DocIDs not sorted at position %d: %d <= %d", 
				i, decoded.Results[i].DocID, decoded.Results[i-1].DocID))
			break
		}
	}

	// Check score ranges
	for i, result := range decoded.Results {
		if result.Score < 0 || result.Score > 1 {
			issues = append(issues, fmt.Sprintf("Invalid score at position %d: %.3f", i, result.Score))
		}
	}

	return issues
}

// exportRankingCacheMetrics exports comprehensive cache metrics for monitoring
func (rc *RankingCache) exportRankingCacheMetrics() map[string]interface{} {
	rc.mu.RLock()
	defer rc.mu.RUnlock()

	metrics := map[string]interface{}{
		"cache_slots": map[string]interface{}{
			"total_slots":     len(rc.slots),
			"used_slots":      0,
			"empty_slots":     0,
			"total_entries":   0,
			"total_bytes":     0,
			"avg_bytes_per_slot": 0.0,
		},
		"registries": map[string]interface{}{
			"summaries_count": len(rc.summaries),
			"urls_count":     len(rc.urls),
		},
		"hash_index": map[string]interface{}{
			"entries": len(rc.hashIndex),
		},
		"slot_utilization": make([]map[string]interface{}, 0),
	}

	var usedSlots, totalEntries, totalBytes int
	var mostUsed, leastUsed *rankingSlot
	var oldestSlot, newestSlot *rankingSlot

	for i, slot := range rc.slots {
		if slot != nil {
			usedSlots++
			totalEntries += slot.meta.Count
			totalBytes += slot.meta.ByteLength

			slotInfo := map[string]interface{}{
				"slot_index":  i,
				"key":         string(rankingAlphabet[i]),
				"hash":        slot.hash,
				"count":       slot.meta.Count,
				"bytes":       slot.meta.ByteLength,
				"used_count":  slot.used,
				"created_at":  slot.meta.CreatedAt,
			}
			
			metrics["slot_utilization"] = append(metrics["slot_utilization"].([]map[string]interface{}), slotInfo)

			// Track usage statistics
			if mostUsed == nil || slot.used > mostUsed.used {
				mostUsed = slot
			}
			if leastUsed == nil || slot.used < leastUsed.used {
				leastUsed = slot
			}
			if oldestSlot == nil || slot.meta.CreatedAt.Before(oldestSlot.meta.CreatedAt) {
				oldestSlot = slot
			}
			if newestSlot == nil || slot.meta.CreatedAt.After(newestSlot.meta.CreatedAt) {
				newestSlot = slot
			}
		}
	}

	// Update slot metrics
	slotMetrics := metrics["cache_slots"].(map[string]interface{})
	slotMetrics["used_slots"] = usedSlots
	slotMetrics["empty_slots"] = len(rc.slots) - usedSlots
	slotMetrics["total_entries"] = totalEntries
	slotMetrics["total_bytes"] = totalBytes
	if usedSlots > 0 {
		slotMetrics["avg_bytes_per_slot"] = float64(totalBytes) / float64(usedSlots)
	}

	// Usage statistics
	if mostUsed != nil && leastUsed != nil {
		metrics["usage_stats"] = map[string]interface{}{
			"most_used_count":  mostUsed.used,
			"least_used_count": leastUsed.used,
			"oldest_slot":      oldestSlot.meta.CreatedAt,
			"newest_slot":      newestSlot.meta.CreatedAt,
		}
	}

	return metrics
}