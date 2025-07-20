export class ContextService {
  static async getCurrentContext() {
    // TODO: Implement context retrieval using Drizzle ORM and pgvector
    return {
      pageType: "dashboard",
      entityId: null,
      userActivity: [],
      recentActions: [],
      complexity: 0.2,
      urgency: 0.1,
    };
  }
}
