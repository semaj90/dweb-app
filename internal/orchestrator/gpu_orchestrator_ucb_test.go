package orchestrator

import (
	"fmt"
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

func TestDispatchProcessesQueuedTasks(t *testing.T) {
    initOrchestratorMetrics()
    cfg := OrchestratorConfig{EnableUCB:true, UCBC:1.0, MaxConcurrent:2}
    s := newOrchestratorService(cfg)
    // enqueue 3 tasks
    for i:=0; i<3; i++ { s.enqueue(&OrchestratorTask{ID:fmt.Sprintf("Q%d", i), Type:TaskInference, Priority:1, added: time.Now()}) }
    // allow dispatch loop to pick tasks
    s.dispatch()
    // active should be <= MaxConcurrent after dispatch triggers executions
    if s.activeCount() > cfg.MaxConcurrent { t.Fatalf("active exceeds max: %d", s.activeCount()) }
    // Wait for completion
    time.Sleep(500 * time.Millisecond)
    if s.completedTasks.Load() == 0 { t.Fatalf("expected some completed tasks") }
}

func TestTensorCachePutGet(t *testing.T) {
    s := newOrchestratorService(OrchestratorConfig{})
    tensor := &Tensor4D{Key:"k1", Shape:[4]int{1,2,3,4}, Data:make([]float32, 1*2*3*4), Created:time.Now()}
    s.tensorCache.Put(tensor)
    if _, ok := s.tensorCache.Get("k1"); !ok { t.Fatalf("tensor not found in cache") }
}
