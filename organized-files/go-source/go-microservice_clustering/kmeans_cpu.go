package clustering

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

// KMeansCPU is a simple, dependency-free CPU implementation.
type KMeansCPU struct{}

func (KMeansCPU) Name() string { return "kmeans" }

func (KMeansCPU) Cluster(data [][]float64, params Params) ([]int, [][]float64, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, nil, errors.New("empty data")
	}
	n, d := len(data), len(data[0])
	params.Validate(4)
	if params.K > n {
		params.K = n
	}

	// Seed RNG
	seed := params.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rnd := rand.New(rand.NewSource(seed))

	// k-means++ init (simplified)
	centroids := make([][]float64, params.K)
	first := rnd.Intn(n)
	centroids[0] = append([]float64(nil), data[first]...)
	dist := make([]float64, n)
	for i := 1; i < params.K; i++ {
		var sum float64
		for j := 0; j < n; j++ {
			// distance to nearest existing centroid
			min := math.MaxFloat64
			for c := 0; c < i; c++ {
				d2 := sqrDist(data[j], centroids[c])
				if d2 < min {
					min = d2
				}
			}
			dist[j] = min
			sum += min
		}
		// weighted pick
		target := rnd.Float64() * sum
		var acc float64
		picked := 0
		for j := 0; j < n; j++ {
			acc += dist[j]
			if acc >= target {
				picked = j
				break
			}
		}
		centroids[i] = append([]float64(nil), data[picked]...)
	}

	assignments := make([]int, n)
	for iter := 0; iter < params.MaxIter; iter++ {
		// Assignment step
		changed := 0
		for i := 0; i < n; i++ {
			bestC := 0
			bestD := math.MaxFloat64
			for c := 0; c < params.K; c++ {
				d2 := sqrDist(data[i], centroids[c])
				if d2 < bestD {
					bestD = d2
					bestC = c
				}
			}
			if assignments[i] != bestC {
				assignments[i] = bestC
				changed++
			}
		}

		// Update step
		newCentroids := make([][]float64, params.K)
		counts := make([]int, params.K)
		for c := 0; c < params.K; c++ {
			newCentroids[c] = make([]float64, d)
		}
		for i := 0; i < n; i++ {
			c := assignments[i]
			counts[c]++
			for j := 0; j < d; j++ {
				newCentroids[c][j] += data[i][j]
			}
		}
		for c := 0; c < params.K; c++ {
			if counts[c] == 0 {
				// re-seed empty cluster
				idx := rnd.Intn(n)
				copy(newCentroids[c], data[idx])
				counts[c] = 1
			} else {
				inv := 1.0 / float64(counts[c])
				for j := 0; j < d; j++ {
					newCentroids[c][j] *= inv
				}
			}
		}

		// Convergence check
		delta := 0.0
		for c := 0; c < params.K; c++ {
			delta += math.Sqrt(sqrDist(centroids[c], newCentroids[c]))
		}
		centroids = newCentroids
		if delta < params.Tolerance {
			break
		}
		if changed == 0 {
			break
		}
	}

	return assignments, centroids, nil
}

func sqrDist(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}
