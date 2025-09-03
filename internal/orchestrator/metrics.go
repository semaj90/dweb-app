package orchestrator

import (
	"net/http"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    orchestratorOnce sync.Once
    taskCounter = prometheus.NewCounterVec(prometheus.CounterOpts{Name: "orchestrator_tasks_total", Help: "Total tasks processed by type and status"}, []string{"type", "status"})
    taskLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "orchestrator_task_latency_ms", Help: "Latency of tasks in ms", Buckets: []float64{10,25,50,75,100,150,250,400,600,800,1000,1500,2500}}, []string{"type", "status"})
)

func initOrchestratorMetrics() { orchestratorOnce.Do(func() { prometheus.MustRegister(taskCounter, taskLatency) }) }
func (s *OrchestratorService) exposePromMetrics(w http.ResponseWriter, r *http.Request) { promhttp.Handler().ServeHTTP(w, r) }
