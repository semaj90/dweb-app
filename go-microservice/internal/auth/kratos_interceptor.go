// kratos_interceptor.go - Enterprise Security and Identity Integration
// Version 2.0 - Kratos authentication and authorization for gRPC services
package auth

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	kratos "github.com/ory/kratos-client-go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// UserIdentity represents authenticated user information
type UserIdentity struct {
	ID          string                 `json:"id"`
	Email       string                 `json:"email"`
	Name        string                 `json:"name"`
	Traits      map[string]interface{} `json:"traits"`
	Roles       []string               `json:"roles"`
	Permissions []string               `json:"permissions"`
	SessionID   string                 `json:"session_id"`
	ExpiresAt   time.Time             `json:"expires_at"`
}

// AuthInterceptorConfig holds authentication configuration
type AuthInterceptorConfig struct {
	KratosPublicURL    string        `json:"kratos_public_url"`
	KratosAdminURL     string        `json:"kratos_admin_url"`
	RequireAuth        bool          `json:"require_auth"`
	AllowedMethods     []string      `json:"allowed_methods"`    // Methods that don't require auth
	RequiredRoles      []string      `json:"required_roles"`     // Global required roles
	SessionTimeout     time.Duration `json:"session_timeout"`
	EnableRBAC         bool          `json:"enable_rbac"`        // Role-based access control
	EnablePermissions  bool          `json:"enable_permissions"` // Permission-based access control
	CacheTimeout       time.Duration `json:"cache_timeout"`      // Session cache timeout
}

// KratosAuthInterceptor provides enterprise authentication and authorization
type KratosAuthInterceptor struct {
	config      AuthInterceptorConfig
	kratosPublic *kratos.APIClient
	kratosAdmin  *kratos.APIClient
	
	// Session cache for performance
	sessionCache map[string]*cachedSession
	
	// RBAC configuration
	rolePermissions map[string][]string // role -> permissions mapping
	methodRoles     map[string][]string // method -> required roles mapping
}

type cachedSession struct {
	identity  *UserIdentity
	createdAt time.Time
}

// NewKratosAuthInterceptor creates a new Kratos authentication interceptor
func NewKratosAuthInterceptor(config AuthInterceptorConfig) *KratosAuthInterceptor {
	// Initialize Kratos public API client
	publicConfig := kratos.NewConfiguration()
	publicConfig.Servers = []kratos.ServerConfiguration{
		{URL: config.KratosPublicURL},
	}
	kratosPublic := kratos.NewAPIClient(publicConfig)
	
	// Initialize Kratos admin API client
	adminConfig := kratos.NewConfiguration()
	adminConfig.Servers = []kratos.ServerConfiguration{
		{URL: config.KratosAdminURL},
	}
	kratosAdmin := kratos.NewAPIClient(adminConfig)
	
	interceptor := &KratosAuthInterceptor{
		config:       config,
		kratosPublic: kratosPublic,
		kratosAdmin:  kratosAdmin,
		sessionCache: make(map[string]*cachedSession),
	}
	
	// Initialize RBAC configuration
	interceptor.initializeRBAC()
	
	// Start cache cleanup routine
	go interceptor.startCacheCleanup()
	
	log.Printf("✅ Kratos Auth Interceptor initialized - Public: %s, Admin: %s", 
		config.KratosPublicURL, config.KratosAdminURL)
	
	return interceptor
}

// UnaryServerInterceptor returns a gRPC unary server interceptor for authentication
func (k *KratosAuthInterceptor) UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		// Check if this method requires authentication
		if !k.requiresAuth(info.FullMethod) {
			return handler(ctx, req)
		}
		
		// Extract and validate session
		identity, err := k.validateRequest(ctx, info.FullMethod)
		if err != nil {
			return nil, err
		}
		
		// Add user identity to context
		ctx = k.addIdentityToContext(ctx, identity)
		
		// Log authentication event
		log.Printf("✅ Authenticated request: user=%s method=%s", identity.ID, info.FullMethod)
		
		return handler(ctx, req)
	}
}

// StreamServerInterceptor returns a gRPC stream server interceptor for authentication
func (k *KratosAuthInterceptor) StreamServerInterceptor() grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		stream grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		// Check if this method requires authentication
		if !k.requiresAuth(info.FullMethod) {
			return handler(srv, stream)
		}
		
		// Extract and validate session
		ctx := stream.Context()
		identity, err := k.validateRequest(ctx, info.FullMethod)
		if err != nil {
			return err
		}
		
		// Create new stream with authenticated context
		wrappedStream := &authenticatedStream{
			ServerStream: stream,
			ctx:          k.addIdentityToContext(ctx, identity),
		}
		
		log.Printf("✅ Authenticated stream: user=%s method=%s", identity.ID, info.FullMethod)
		
		return handler(srv, wrappedStream)
	}
}

type authenticatedStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (s *authenticatedStream) Context() context.Context {
	return s.ctx
}

// validateRequest performs the complete authentication and authorization flow
func (k *KratosAuthInterceptor) validateRequest(ctx context.Context, method string) (*UserIdentity, error) {
	// Extract session information from metadata
	sessionToken, err := k.extractSessionToken(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Unauthenticated, "Authentication required: %v", err)
	}
	
	// Check session cache first
	if cachedSession := k.getFromCache(sessionToken); cachedSession != nil {
		// Validate cached session is still valid
		if time.Since(cachedSession.createdAt) < k.config.CacheTimeout {
			// Check authorization for this specific method
			if err := k.checkAuthorization(cachedSession.identity, method); err != nil {
				return nil, err
			}
			return cachedSession.identity, nil
		}
		// Remove expired cache entry
		delete(k.sessionCache, sessionToken)
	}
	
	// Validate session with Kratos
	identity, err := k.validateSessionWithKratos(ctx, sessionToken)
	if err != nil {
		return nil, status.Errorf(codes.Unauthenticated, "Invalid session: %v", err)
	}
	
	// Check authorization
	if err := k.checkAuthorization(identity, method); err != nil {
		return nil, err
	}
	
	// Cache the session
	k.cacheSession(sessionToken, identity)
	
	return identity, nil
}

// extractSessionToken extracts session information from gRPC metadata
func (k *KratosAuthInterceptor) extractSessionToken(ctx context.Context) (string, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return "", fmt.Errorf("metadata not provided")
	}
	
	// Try to get session token from various headers
	var sessionToken string
	
	// Method 1: Authorization header (Bearer token)
	if values := md.Get("authorization"); len(values) > 0 {
		auth := values[0]
		if strings.HasPrefix(auth, "Bearer ") {
			sessionToken = strings.TrimPrefix(auth, "Bearer ")
		}
	}
	
	// Method 2: X-Session-Token header
	if sessionToken == "" {
		if values := md.Get("x-session-token"); len(values) > 0 {
			sessionToken = values[0]
		}
	}
	
	// Method 3: Cookie header (for web clients)
	if sessionToken == "" {
		if values := md.Get("cookie"); len(values) > 0 {
			cookies := values[0]
			// Parse cookies and extract session
			if token := k.extractSessionFromCookies(cookies); token != "" {
				sessionToken = token
			}
		}
	}
	
	if sessionToken == "" {
		return "", fmt.Errorf("session token not found in request")
	}
	
	return sessionToken, nil
}

// extractSessionFromCookies parses cookies to find session token
func (k *KratosAuthInterceptor) extractSessionFromCookies(cookies string) string {
	// Simple cookie parsing - in production, use a proper cookie parser
	parts := strings.Split(cookies, ";")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if strings.HasPrefix(part, "ory_kratos_session=") {
			return strings.TrimPrefix(part, "ory_kratos_session=")
		}
	}
	return ""
}

// validateSessionWithKratos validates the session with Kratos backend
func (k *KratosAuthInterceptor) validateSessionWithKratos(ctx context.Context, sessionToken string) (*UserIdentity, error) {
	// Create HTTP request context with cookie
	req := k.kratosPublic.FrontendApi.ToSession(ctx)
	
	// Add session cookie
	cookieHeader := fmt.Sprintf("ory_kratos_session=%s", sessionToken)
	req = req.Cookie(cookieHeader)
	
	// Execute session validation
	session, httpResp, err := req.Execute()
	if err != nil {
		return nil, fmt.Errorf("kratos session validation failed: %w", err)
	}
	
	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("kratos session validation returned status %d", httpResp.StatusCode)
	}
	
	if session == nil || !*session.Active {
		return nil, fmt.Errorf("session is not active")
	}
	
	// Extract user identity information
	identity := &UserIdentity{
		ID:        session.Identity.Id,
		SessionID: session.Id,
		Traits:    make(map[string]interface{}),
	}
	
	// Extract traits (email, name, etc.)
	if session.Identity.Traits != nil {
		if email, ok := session.Identity.Traits["email"].(string); ok {
			identity.Email = email
		}
		if name, ok := session.Identity.Traits["name"].(string); ok {
			identity.Name = name
		}
		identity.Traits = session.Identity.Traits
	}
	
	// Set expiration time
	if session.ExpiresAt != nil {
		identity.ExpiresAt = *session.ExpiresAt
	} else {
		// Default to session timeout if not specified
		identity.ExpiresAt = time.Now().Add(k.config.SessionTimeout)
	}
	
	// Extract roles and permissions from identity metadata
	identity.Roles = k.extractRolesFromIdentity(session.Identity)
	identity.Permissions = k.extractPermissionsFromRoles(identity.Roles)
	
	return identity, nil
}

// checkAuthorization validates user authorization for the requested method
func (k *KratosAuthInterceptor) checkAuthorization(identity *UserIdentity, method string) error {
	// Check global required roles
	if len(k.config.RequiredRoles) > 0 {
		if !k.hasAnyRole(identity.Roles, k.config.RequiredRoles) {
			return status.Errorf(codes.PermissionDenied, 
				"User lacks required global roles: %v", k.config.RequiredRoles)
		}
	}
	
	// Check method-specific roles if RBAC is enabled
	if k.config.EnableRBAC {
		if requiredRoles, exists := k.methodRoles[method]; exists {
			if !k.hasAnyRole(identity.Roles, requiredRoles) {
				return status.Errorf(codes.PermissionDenied, 
					"User lacks required roles for %s: %v", method, requiredRoles)
			}
		}
	}
	
	// Check permissions if permission-based access control is enabled
	if k.config.EnablePermissions {
		requiredPermission := k.getRequiredPermissionForMethod(method)
		if requiredPermission != "" {
			if !k.hasPermission(identity.Permissions, requiredPermission) {
				return status.Errorf(codes.PermissionDenied, 
					"User lacks required permission for %s: %s", method, requiredPermission)
			}
		}
	}
	
	return nil
}

// Helper methods for RBAC and permissions
func (k *KratosAuthInterceptor) hasAnyRole(userRoles, requiredRoles []string) bool {
	for _, userRole := range userRoles {
		for _, requiredRole := range requiredRoles {
			if userRole == requiredRole {
				return true
			}
		}
	}
	return false
}

func (k *KratosAuthInterceptor) hasPermission(userPermissions []string, requiredPermission string) bool {
	for _, permission := range userPermissions {
		if permission == requiredPermission {
			return true
		}
	}
	return false
}

func (k *KratosAuthInterceptor) extractRolesFromIdentity(identity *kratos.Identity) []string {
	// Extract roles from identity metadata - customize based on your Kratos setup
	roles := []string{"user"} // Default role
	
	if identity.MetadataPublic != nil {
		if rolesInterface, ok := identity.MetadataPublic["roles"]; ok {
			if rolesList, ok := rolesInterface.([]interface{}); ok {
				for _, role := range rolesList {
					if roleStr, ok := role.(string); ok {
						roles = append(roles, roleStr)
					}
				}
			}
		}
	}
	
	return roles
}

func (k *KratosAuthInterceptor) extractPermissionsFromRoles(roles []string) []string {
	var permissions []string
	for _, role := range roles {
		if rolePerms, exists := k.rolePermissions[role]; exists {
			permissions = append(permissions, rolePerms...)
		}
	}
	
	// Remove duplicates
	permSet := make(map[string]bool)
	var uniquePerms []string
	for _, perm := range permissions {
		if !permSet[perm] {
			permSet[perm] = true
			uniquePerms = append(uniquePerms, perm)
		}
	}
	
	return uniquePerms
}

func (k *KratosAuthInterceptor) getRequiredPermissionForMethod(method string) string {
	// Map gRPC methods to required permissions
	methodPermissions := map[string]string{
		"/aiserver.VectorService/ProcessRotation":        "vector.process",
		"/aiserver.VectorService/ProcessSimilarity":      "vector.similarity",
		"/aiserver.VectorService/ProcessLegalDocument":   "document.process",
		"/aiserver.VectorService/FindSimilarDocuments":   "document.search",
		"/aiserver.VectorService/BatchProcessVectors":    "vector.batch",
		"/aiserver.AsyncJobService/SubmitAsyncJob":       "job.submit",
		"/aiserver.AsyncJobService/GetJobStatus":         "job.read",
		"/aiserver.AsyncJobService/GetJobResult":         "job.read",
		"/aiserver.AsyncJobService/CancelJob":            "job.cancel",
	}
	
	return methodPermissions[method]
}

// Cache management
func (k *KratosAuthInterceptor) cacheSession(sessionToken string, identity *UserIdentity) {
	k.sessionCache[sessionToken] = &cachedSession{
		identity:  identity,
		createdAt: time.Now(),
	}
}

func (k *KratosAuthInterceptor) getFromCache(sessionToken string) *cachedSession {
	return k.sessionCache[sessionToken]
}

func (k *KratosAuthInterceptor) startCacheCleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		now := time.Now()
		for token, cached := range k.sessionCache {
			if now.Sub(cached.createdAt) > k.config.CacheTimeout {
				delete(k.sessionCache, token)
			}
		}
	}
}

// Configuration helpers
func (k *KratosAuthInterceptor) requiresAuth(method string) bool {
	if !k.config.RequireAuth {
		return false
	}
	
	// Check if method is in allowed (no-auth) list
	for _, allowedMethod := range k.config.AllowedMethods {
		if method == allowedMethod {
			return false
		}
	}
	
	return true
}

func (k *KratosAuthInterceptor) initializeRBAC() {
	// Initialize role-to-permissions mapping
	k.rolePermissions = map[string][]string{
		"admin": {
			"vector.process", "vector.similarity", "vector.batch",
			"document.process", "document.search", "document.manage",
			"job.submit", "job.read", "job.cancel", "job.manage",
			"system.monitor", "system.configure",
		},
		"legal_professional": {
			"vector.process", "vector.similarity",
			"document.process", "document.search",
			"job.submit", "job.read", "job.cancel",
		},
		"paralegal": {
			"document.search", "job.read",
		},
		"user": {
			"document.search",
		},
	}
	
	// Initialize method-to-roles mapping
	k.methodRoles = map[string][]string{
		"/aiserver.VectorService/BatchProcessVectors":    {"admin"},
		"/aiserver.VectorService/GetMetrics":             {"admin"},
		"/aiserver.AsyncJobService/CancelJob":            {"admin", "legal_professional"},
	}
}

// Utility functions for extracting user identity from context
func GetUserIdentityFromContext(ctx context.Context) (*UserIdentity, bool) {
	identity, ok := ctx.Value("user_identity").(*UserIdentity)
	return identity, ok
}

func GetUserIDFromContext(ctx context.Context) (string, bool) {
	identity, ok := GetUserIdentityFromContext(ctx)
	if !ok {
		return "", false
	}
	return identity.ID, true
}

func (k *KratosAuthInterceptor) addIdentityToContext(ctx context.Context, identity *UserIdentity) context.Context {
	ctx = context.WithValue(ctx, "user_identity", identity)
	ctx = context.WithValue(ctx, "user_id", identity.ID)
	ctx = context.WithValue(ctx, "user_roles", identity.Roles)
	ctx = context.WithValue(ctx, "user_permissions", identity.Permissions)
	return ctx
}