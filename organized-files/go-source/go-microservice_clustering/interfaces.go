package clustering

// Algorithm is the interface for pluggable clustering algorithms.
type Algorithm interface {
	// Name returns a short identifier for the algorithm (e.g., "kmeans").
	Name() string
	// Cluster clusters the given data matrix [n x d] into k clusters.
	// Returns the cluster assignment for each row, the centroids [k x d], and an error.
	Cluster(data [][]float64, params Params) (assignments []int, centroids [][]float64, err error)
}

// Params captures generic clustering parameters.
type Params struct {
	K         int     // number of clusters
	MaxIter   int     // maximum iterations
	Tolerance float64 // convergence threshold
	Seed      int64   // RNG seed (optional)
}

// Validate applies basic sanity checks and defaulting.
func (p *Params) Validate(defaultK int) {
	if p.K <= 0 {
		p.K = defaultK
	}
	if p.MaxIter <= 0 {
		p.MaxIter = 100
	}
	if p.Tolerance <= 0 {
		p.Tolerance = 1e-4
	}
}
