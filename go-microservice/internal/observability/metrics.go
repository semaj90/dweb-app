package observability

import (
	"context"
	"sync"
	"time"
)

// RequestCounter is a lightweight in-memory counter for tracking request lifecycle metrics.
type RequestCounter struct {
	mu        sync.Mutex
	requests  map[string]int64
	successes map[string]int64
	errors    map[string]map[string]int64 // op -> errorCode -> count
	updated   time.Time
}

// NewRequestCounter creates a new RequestCounter
func NewRequestCounter() *RequestCounter {
	return &RequestCounter{
		requests:  make(map[string]int64),
		successes: make(map[string]int64),
		errors:    make(map[string]map[string]int64),
		updated:   time.Now(),
	}
}

// IncrementRequest records a new request attempt for the given operation
func (r *RequestCounter) IncrementRequest(op string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests[op]++
	r.updated = time.Now()
}

// IncrementSuccess records a successful operation
func (r *RequestCounter) IncrementSuccess(op string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.successes[op]++
	r.updated = time.Now()
}

// IncrementError records a failure for the operation with an error code
func (r *RequestCounter) IncrementError(op, code string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.errors[op]; !ok {
		r.errors[op] = make(map[string]int64)
	}
	r.errors[op][code]++
	r.updated = time.Now()
}

// Snapshot returns a copy of current counters
func (r *RequestCounter) Snapshot() (requests, successes map[string]int64, errors map[string]map[string]int64, updated time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	reqCopy := make(map[string]int64, len(r.requests))
	for k, v := range r.requests {
		reqCopy[k] = v
	}
	succCopy := make(map[string]int64, len(r.successes))
	for k, v := range r.successes {
		succCopy[k] = v
	}
	errCopy := make(map[string]map[string]int64, len(r.errors))
	for k, m := range r.errors {
		mm := make(map[string]int64, len(m))
		for kk, vv := range m {
			mm[kk] = vv
		}
		errCopy[k] = mm
	}
	return reqCopy, succCopy, errCopy, r.updated
}

// HealthChecker is a small helper that periodically emits health metrics using the ELKLogger.
type HealthChecker struct {
	logger   *ELKLogger
	interval time.Duration
	stopCh   chan struct{}
	stopped  chan struct{}
}

// NewHealthChecker creates a new HealthChecker bound to the provided logger.
// Interval defaults to 30s if zero.
func NewHealthChecker(logger *ELKLogger) *HealthChecker {
	interval := 30 * time.Second
	return &HealthChecker{
		logger:   logger,
		interval: interval,
		stopCh:   make(chan struct{}),
		stopped:  make(chan struct{}),
	}
}

// Start begins periodic health emissions. Call in a goroutine or let caller manage lifecycle.
func (h *HealthChecker) Start(ctx context.Context) {
	ticker := time.NewTicker(h.interval)
	defer func() {
		ticker.Stop()
		close(h.stopped)
	}()

	for {
		select {
		case <-ctx.Done():
			h.logger.Info("HealthChecker stopping due to context cancellation").Send()
			return
		case <-h.stopCh:
			h.logger.Info("HealthChecker stopped").Send()
			return
		case <-ticker.C:
			// Emit a simple health tick
			h.logger.Info("HealthChecker tick").
				Field("time", time.Now().UTC().Format(time.RFC3339)).
				Send()
		}
	}
}

// Stop requests the health checker to stop and waits for it to finish.
func (h *HealthChecker) Stop() {
	select {
	case <-h.stopped:
		return
	default:
	}
	close(h.stopCh)
	<-h.stopped
}
