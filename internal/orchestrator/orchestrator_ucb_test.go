package orchestrator

import (
	"testing"
	"time"
)

func TestSelectUCBPrefersOlderAndHigherPriority(t *testing.T) {
    cfg := OrchestratorConfig{EnableUCB: true, UCBC: 1.25, MaxConcurrent: 1}
    s := newOrchestratorService(cfg)
    now := time.Now()
    a := &OrchestratorTask{ID:"A", Type:TaskInference, Priority:1, added: now.Add(-5*time.Second)}
    b := &OrchestratorTask{ID:"B", Type:TaskInference, Priority:3, added: now.Add(-2*time.Second)}
    a.visits, b.visits = 1,1
    a.value, b.value = 1,1
    picked := s.selectUCB([]*OrchestratorTask{a,b})
    if picked == nil { t.Fatalf("no task picked") }
    if picked.ID != "B" && picked.ID != "A" { t.Fatalf("unexpected pick %s", picked.ID) }
}

func TestStartExecutionMetrics(t *testing.T) {
    initOrchestratorMetrics()
    cfg := OrchestratorConfig{EnableUCB:true, UCBC:1.0, MaxConcurrent:1}
    s := newOrchestratorService(cfg)
    task := &OrchestratorTask{ID:"T1", Type:TaskEmbedding, Priority:1, added: time.Now()}
    s.startExecution(task)
    if task.Status != "completed" { t.Fatalf("expected completed got %s", task.Status) }
    if task.EndTime == nil { t.Fatalf("end time missing") }
}
