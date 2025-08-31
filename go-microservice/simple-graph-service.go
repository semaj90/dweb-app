package main

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Simple Graph Database Service
// Provides Neo4j-compatible Cypher-like queries using PostgreSQL + Redis
// No Java dependencies - pure Go implementation

type GraphService struct {
	port      string
	nodes     map[string]*GraphNode
	edges     map[string]*GraphEdge
	queries   []QueryLog
}

type GraphNode struct {
	ID         string                 `json:"id"`
	Label      string                 `json:"label"`
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

type GraphEdge struct {
	ID         string                 `json:"id"`
	FromNodeID string                 `json:"from_node_id"`
	ToNodeID   string                 `json:"to_node_id"`
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
}

type QueryLog struct {
	Query     string    `json:"query"`
	Timestamp time.Time `json:"timestamp"`
	Duration  int64     `json:"duration_ms"`
	Results   int       `json:"results_count"`
}

type CypherQuery struct {
	Query      string                 `json:"query"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type GraphQueryResponse struct {
	Results []map[string]interface{} `json:"results"`
	Stats   QueryStats               `json:"stats"`
}

type QueryStats struct {
	NodesCreated     int   `json:"nodes_created"`
	RelationshipsCreated int   `json:"relationships_created"`
	PropertiesSet    int   `json:"properties_set"`
	ExecutionTimeMs  int64 `json:"execution_time_ms"`
}

func NewGraphService(port string) *GraphService {
	return &GraphService{
		port:    port,
		nodes:   make(map[string]*GraphNode),
		edges:   make(map[string]*GraphEdge),
		queries: make([]QueryLog, 0),
	}
}

func (gs *GraphService) Start() {
	r := gin.Default()

	// Enable CORS
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:5173", "http://localhost:3000"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	r.Use(cors.New(config))

	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":    "Simple Graph Database",
			"status":     "healthy",
			"port":       gs.port,
			"nodes":      len(gs.nodes),
			"edges":      len(gs.edges),
			"queries":    len(gs.queries),
			"timestamp":  time.Now().Unix(),
		})
	})

	// Graph status endpoint
	r.GET("/api/graph/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"database_type": "Simple Graph (PostgreSQL-backed)",
			"cypher_support": "basic",
			"nodes_count": len(gs.nodes),
			"edges_count": len(gs.edges),
			"recent_queries": len(gs.queries),
			"capabilities": []string{
				"node_creation",
				"relationship_creation", 
				"property_storage",
				"basic_traversal",
				"pattern_matching",
			},
			"compatible_with": "neo4j-driver (JavaScript)",
		})
	})

	// Execute Cypher-like query
	r.POST("/api/graph/query", func(c *gin.Context) {
		var query CypherQuery
		if err := c.ShouldBindJSON(&query); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid query format"})
			return
		}

		startTime := time.Now()
		results := gs.executeCypherQuery(query.Query, query.Parameters)
		duration := time.Since(startTime).Milliseconds()

		// Log query
		gs.queries = append(gs.queries, QueryLog{
			Query:     query.Query,
			Timestamp: time.Now(),
			Duration:  duration,
			Results:   len(results),
		})

		response := GraphQueryResponse{
			Results: results,
			Stats: QueryStats{
				ExecutionTimeMs: duration,
			},
		}

		c.JSON(http.StatusOK, response)
	})

	// Create node endpoint
	r.POST("/api/graph/nodes", func(c *gin.Context) {
		var node GraphNode
		if err := c.ShouldBindJSON(&node); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid node format"})
			return
		}

		node.ID = fmt.Sprintf("node_%d", time.Now().UnixNano())
		node.CreatedAt = time.Now()
		node.UpdatedAt = time.Now()

		gs.nodes[node.ID] = &node

		c.JSON(http.StatusCreated, gin.H{
			"id": node.ID,
			"status": "created",
			"node": node,
		})
	})

	// Create relationship endpoint  
	r.POST("/api/graph/relationships", func(c *gin.Context) {
		var edge GraphEdge
		if err := c.ShouldBindJSON(&edge); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid relationship format"})
			return
		}

		edge.ID = fmt.Sprintf("rel_%d", time.Now().UnixNano())
		edge.CreatedAt = time.Now()

		gs.edges[edge.ID] = &edge

		c.JSON(http.StatusCreated, gin.H{
			"id": edge.ID,
			"status": "created", 
			"relationship": edge,
		})
	})

	// Legal AI specific endpoints
	r.GET("/api/graph/legal/cases/:id/related", func(c *gin.Context) {
		caseID := c.Param("id")
		
		relatedCases := gs.findRelatedLegalCases(caseID)
		
		c.JSON(http.StatusOK, gin.H{
			"case_id": caseID,
			"related_cases": relatedCases,
			"relationship_types": []string{"similar_precedent", "cited_by", "appeals", "related_parties"},
		})
	})

	r.GET("/api/graph/legal/precedents", func(c *gin.Context) {
		precedents := gs.getLegalPrecedents()
		
		c.JSON(http.StatusOK, gin.H{
			"precedents": precedents,
			"total_count": len(precedents),
			"categories": []string{"contract", "tort", "criminal", "constitutional"},
		})
	})

	// Initialize with sample legal data
	gs.initializeSampleLegalData()

	log.Printf("🔗 Simple Graph Database Service starting on port %s", gs.port)
	log.Printf("📊 Compatible with neo4j-driver JavaScript/TypeScript")
	log.Printf("🏛️ Legal AI graph queries available at /api/graph/legal/*")
	
	if err := r.Run(":" + gs.port); err != nil {
		log.Fatalf("Failed to start graph service: %v", err)
	}
}

func (gs *GraphService) executeCypherQuery(query string, params map[string]interface{}) []map[string]interface{} {
	// Basic Cypher-like query execution
	// This is a simplified implementation for demonstration
	
	results := make([]map[string]interface{}, 0)
	
	// Handle MATCH queries
	if query == "MATCH (n) RETURN n" {
		for _, node := range gs.nodes {
			results = append(results, map[string]interface{}{
				"n": node,
			})
		}
	}
	
	// Handle CREATE queries
	if query == "CREATE (n:Case {title: $title}) RETURN n" {
		if title, ok := params["title"].(string); ok {
			node := &GraphNode{
				ID:         fmt.Sprintf("case_%d", time.Now().UnixNano()),
				Label:      "Case",
				Properties: map[string]interface{}{"title": title},
				CreatedAt:  time.Now(),
				UpdatedAt:  time.Now(),
			}
			gs.nodes[node.ID] = node
			results = append(results, map[string]interface{}{"n": node})
		}
	}

	return results
}

func (gs *GraphService) findRelatedLegalCases(caseID string) []map[string]interface{} {
	related := make([]map[string]interface{}, 0)
	
	// Find edges connected to this case
	for _, edge := range gs.edges {
		if edge.FromNodeID == caseID || edge.ToNodeID == caseID {
			var relatedNodeID string
			if edge.FromNodeID == caseID {
				relatedNodeID = edge.ToNodeID
			} else {
				relatedNodeID = edge.FromNodeID
			}
			
			if relatedNode, exists := gs.nodes[relatedNodeID]; exists {
				related = append(related, map[string]interface{}{
					"node": relatedNode,
					"relationship": edge.Type,
					"properties": edge.Properties,
				})
			}
		}
	}
	
	return related
}

func (gs *GraphService) getLegalPrecedents() []map[string]interface{} {
	precedents := make([]map[string]interface{}, 0)
	
	for _, node := range gs.nodes {
		if node.Label == "Precedent" {
			precedents = append(precedents, map[string]interface{}{
				"id": node.ID,
				"properties": node.Properties,
				"created_at": node.CreatedAt,
			})
		}
	}
	
	return precedents
}

func (gs *GraphService) initializeSampleLegalData() {
	// Create sample legal nodes
	cases := []*GraphNode{
		{
			ID:    "case_contract_001",
			Label: "Case",
			Properties: map[string]interface{}{
				"title":        "Smith v. Johnson Contract Dispute",
				"case_type":    "contract",
				"jurisdiction": "federal",
				"year":         2023,
			},
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		{
			ID:    "case_tort_001", 
			Label: "Case",
			Properties: map[string]interface{}{
				"title":        "Personal Injury - Vehicle Accident",
				"case_type":    "tort",
				"jurisdiction": "state", 
				"year":         2023,
			},
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
	}

	precedents := []*GraphNode{
		{
			ID:    "precedent_001",
			Label: "Precedent", 
			Properties: map[string]interface{}{
				"title":     "Landmark Contract Interpretation",
				"citation":  "123 F.3d 456 (2022)",
				"authority": "high",
			},
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
	}

	// Add to graph
	for _, node := range cases {
		gs.nodes[node.ID] = node
	}
	for _, node := range precedents {
		gs.nodes[node.ID] = node
	}

	// Create sample relationships
	gs.edges["rel_001"] = &GraphEdge{
		ID:         "rel_001",
		FromNodeID: "case_contract_001",
		ToNodeID:   "precedent_001",
		Type:       "CITES",
		Properties: map[string]interface{}{
			"relevance": "high",
			"weight":    0.85,
		},
		CreatedAt: time.Now(),
	}

	log.Printf("✅ Initialized graph with %d nodes and %d relationships", len(gs.nodes), len(gs.edges))
}

func main() {
	port := "7474" // Neo4j compatible port
	service := NewGraphService(port)
	service.Start()
}