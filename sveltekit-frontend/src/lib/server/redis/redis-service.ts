import { createClient } from "redis";

// Redis pub/sub service for real-time updates

export interface RedisConfig {
  url: string;
  // retryDelayOnFailover removed - deprecated in newer redis versions
  maxRetriesPerRequest?: number;
}
class RedisService {
  private client: any;
  private publisher: any;
  private subscriber: any;
  private isConnected = false;
  private realTimeServer: any;

  constructor() {
    // Remove circular dependency - will be set externally if needed
    this.initializeClients();
  }
  private async initializeClients() {
    const config: RedisConfig = {
      url: import.meta.env.REDIS_URL || "redis://localhost:6379",
      // retryDelayOnFailover: 100, // removed - deprecated
      maxRetriesPerRequest: 3,
    };

    try {
      // Main client for operations
      this.client = createClient(config);
      this.publisher = createClient(config);
      this.subscriber = createClient(config);

      // Setup error handlers
      this.client.on("error", this.handleError.bind(this));
      this.publisher.on("error", this.handleError.bind(this));
      this.subscriber.on("error", this.handleError.bind(this));

      // Connect all clients
      await Promise.all([
        this.client.connect(),
        this.publisher.connect(),
        this.subscriber.connect(),
      ]);

      this.isConnected = true;
      console.log("✅ Redis clients connected successfully");
    } catch (error: any) {
      console.error("❌ Redis connection failed:", error);
      this.isConnected = false;
    }
  }
  private handleError(error: Error) {
    console.error("Redis error:", error);
    this.isConnected = false;
  }
  // Evidence Updates
  public async publishEvidenceCreated(
    evidenceId: string,
    evidenceData: any,
    userId?: string,
  ) {
    await this.publish("evidence_update", {
      type: "EVIDENCE_CREATED",
      evidenceId,
      data: evidenceData,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  public async publishEvidenceUpdated(
    evidenceId: string,
    changes: any,
    userId?: string,
  ) {
    await this.publish("evidence_update", {
      type: "EVIDENCE_UPDATED",
      evidenceId,
      changes,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  public async publishEvidenceDeleted(evidenceId: string, userId?: string) {
    await this.publish("evidence_update", {
      type: "EVIDENCE_DELETED",
      evidenceId,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  // Case Updates
  public async publishCaseUpdated(
    caseId: string,
    changes: any,
    userId?: string,
  ) {
    await this.publish("case_update", {
      type: "CASE_UPDATED",
      caseId,
      changes,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  public async publishCaseStatusChanged(
    caseId: string,
    oldStatus: string,
    newStatus: string,
    userId?: string,
  ) {
    await this.publish("case_update", {
      type: "CASE_STATUS_CHANGED",
      caseId,
      oldStatus,
      newStatus,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  // Canvas Updates
  public async publishCanvasNodeMoved(
    caseId: string,
    nodeId: string,
    position: { x: number; y: number },
    userId?: string,
  ) {
    await this.publish("canvas_update", {
      type: "CANVAS_NODE_MOVED",
      caseId,
      nodeId,
      position,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  public async publishCanvasNodeAdded(
    caseId: string,
    nodeData: any,
    userId?: string,
  ) {
    await this.publish("canvas_update", {
      type: "CANVAS_NODE_ADDED",
      caseId,
      nodeData,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  public async publishCanvasStateChanged(
    caseId: string,
    state: any,
    userId?: string,
  ) {
    await this.publish("canvas_update", {
      type: "CANVAS_STATE_CHANGED",
      caseId,
      state,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  // POI Updates
  public async publishPOIUpdated(poiId: string, changes: any, userId?: string) {
    await this.publish("poi_update", {
      type: "POI_UPDATED",
      poiId,
      changes,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  // Report Updates
  public async publishReportUpdated(
    reportId: string,
    changes: any,
    userId?: string,
  ) {
    await this.publish("report_update", {
      type: "REPORT_UPDATED",
      reportId,
      changes,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  // User Activity
  public async publishUserActivity(
    userId: string,
    activity: string,
    metadata?: unknown,
  ) {
    await this.publish("user_activity", {
      type: "USER_ACTIVITY",
      userId,
      activity,
      metadata,
      timestamp: new Date().toISOString(),
    });
  }
  // Generic publish method
  private async publish(channel: string, data: any) {
    if (!this.isConnected) {
      console.warn("Redis not connected, skipping publish");
      return;
    }
    try {
      const message = JSON.stringify(data);
      await this.publisher.publish(channel, message);
    } catch (error: any) {
      console.error(`Failed to publish to ${channel}:`, error);
    }
  }
  // Cache operations
  public async setCache(key: string, value: any, ttlSeconds: number = 300) {
    if (!this.isConnected) return;

    try {
      const serialized = JSON.stringify(value);
      await this.client.setEx(key, ttlSeconds, serialized);
    } catch (error: any) {
      console.error("Cache set error:", error);
    }
  }
  public async getCache(key: string) {
    if (!this.isConnected) return null;

    try {
      const cached = await this.client.get(key);
      return cached ? JSON.parse(cached) : null;
    } catch (error: any) {
      console.error("Cache get error:", error);
      return null;
    }
  }
  public async deleteCache(key: string) {
    if (!this.isConnected) return;

    try {
      await this.client.del(key);
    } catch (error: any) {
      console.error("Cache delete error:", error);
    }
  }
  // Bulk operations
  public async publishBulkEvidenceUpdate(
    evidenceIds: string[],
    action: string,
    userId?: string,
  ) {
    await this.publish("evidence_update", {
      type: "EVIDENCE_BULK_UPDATE",
      evidenceIds,
      action,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  // Analytics and metrics
  public async trackEvent(event: string, data: any, userId?: string) {
    await this.publish("analytics", {
      event,
      data,
      userId,
      timestamp: new Date().toISOString(),
    });
  }
  public isConnectedToRedis(): boolean {
    return this.isConnected;
  }
  public async disconnect() {
    if (this.client) await this.client.disconnect();
    if (this.publisher) await this.publisher.disconnect();
    if (this.subscriber) await this.subscriber.disconnect();
    this.isConnected = false;
  }
}
// Singleton instance
let redisServiceInstance: RedisService | null = null;

export function getRedisService(): RedisService {
  if (!redisServiceInstance) {
    redisServiceInstance = new RedisService();
  }
  return redisServiceInstance;
}
export default RedisService;
