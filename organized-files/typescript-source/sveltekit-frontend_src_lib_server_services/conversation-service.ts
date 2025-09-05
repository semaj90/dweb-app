/**
 * Conversation Service - PostgreSQL Integration
 * Handles AI chat conversation persistence and retrieval
 */

import { db } from '$lib/server/db/pg';
import { conversations, messages as messagesTable } from '$lib/server/db/schema-postgres-enhanced';
import type { 
  Conversation, 
  Message, 
  NewConversation, 
  NewMessage 
} from '$lib/server/db/schema-postgres-enhanced';
import { eq, desc, and, isNull } from 'drizzle-orm';

export class ConversationService {
  /**
   * Create a new conversation
   */
  async createConversation(data: {
    userId: string;
    title: string;
    caseId?: string;
    context?: Record<string, any>;
  }): Promise<Conversation> {
    const newConversation: NewConversation = {
      userId: data.userId,
      title: data.title,
      caseId: data.caseId || null,
      context: data.context || {},
      metadata: {}
    };

    const [conversation] = await db
      .insert(conversations)
      .values(newConversation)
      .returning();

    return conversation;
  }

  /**
   * Get conversation by ID with messages
   */
  async getConversationWithMessages(conversationId: string): Promise<{
    conversation: Conversation;
    messages: Message[];
  } | null> {
    const [conversation] = await db
      .select()
      .from(conversations)
      .where(and(
        eq(conversations.id, conversationId),
        isNull(conversations.archivedAt)
      ))
      .limit(1);

    if (!conversation) {
      return null;
    }

    const messages = await db
      .select()
      .from(messagesTable)
      .where(eq(messagesTable.conversationId, conversationId))
      .orderBy(messagesTable.createdAt);

    return { conversation, messages };
  }

  /**
   * Get user conversations (recent first)
   */
  async getUserConversations(userId: string, limit: number = 50): Promise<Conversation[]> {
    return await db
      .select()
      .from(conversations)
      .where(and(
        eq(conversations.userId, userId),
        isNull(conversations.archivedAt)
      ))
      .orderBy(desc(conversations.updatedAt))
      .limit(limit);
  }

  /**
   * Add message to conversation
   */
  async addMessage(data: {
    conversationId: string;
    role: 'user' | 'assistant';
    content: string;
    model?: string;
    tokenCount?: number;
    processingTime?: number;
    confidence?: number;
    vectorSearchResults?: any[];
    metadata?: Record<string, any>;
  }): Promise<Message> {
    const newMessage: NewMessage = {
      conversationId: data.conversationId,
      role: data.role,
      content: data.content,
      metadata: {
        model: data.model,
        tokenCount: data.tokenCount,
        processingTime: data.processingTime,
        confidence: data.confidence,
        vectorSearchResults: data.vectorSearchResults,
        ...data.metadata
      }
    };

    const [message] = await db
      .insert(messagesTable)
      .values(newMessage)
      .returning();

    // Update conversation updated_at timestamp
    await db
      .update(conversations)
      .set({ updatedAt: new Date() })
      .where(eq(conversations.id, data.conversationId));

    return message;
  }

  /**
   * Update conversation title
   */
  async updateConversationTitle(conversationId: string, title: string): Promise<void> {
    await db
      .update(conversations)
      .set({ 
        title,
        updatedAt: new Date()
      })
      .where(eq(conversations.id, conversationId));
  }

  /**
   * Archive conversation
   */
  async archiveConversation(conversationId: string): Promise<void> {
    await db
      .update(conversations)
      .set({ 
        archivedAt: new Date(),
        updatedAt: new Date()
      })
      .where(eq(conversations.id, conversationId));
  }

  /**
   * Generate conversation title from first message
   */
  generateConversationTitle(firstMessage: string): string {
    const cleaned = firstMessage.replace(/[^\w\s]/gi, '').trim();
    const words = cleaned.split(/\s+/).slice(0, 6);
    let title = words.join(' ');
    
    if (title.length > 50) {
      title = title.substring(0, 47) + '...';
    }
    
    return title || 'New Conversation';
  }

  /**
   * Convert database messages to chat format
   */
  convertTochatMessages(dbMessages: Message[]): Array<{
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    metadata?: Record<string, any>;
  }> {
    return dbMessages.map(msg => ({
      id: msg.id,
      role: msg.role as 'user' | 'assistant' | 'system',
      content: msg.content,
      metadata: msg.metadata
    }));
  }

  /**
   * Get conversation analytics for user
   */
  async getConversationAnalytics(userId: string): Promise<{
    totalConversations: number;
    totalMessages: number;
    averageMessagesPerConversation: number;
    mostActiveDay: string;
  }> {
    // This would require more complex queries - simplified for now
    const userConversations = await this.getUserConversations(userId, 1000);
    
    const totalConversations = userConversations.length;
    
    // Get total messages count (simplified)
    const totalMessages = 0; // Would need aggregation query
    
    return {
      totalConversations,
      totalMessages,
      averageMessagesPerConversation: totalMessages / (totalConversations || 1),
      mostActiveDay: new Date().toISOString().split('T')[0]
    };
  }
}

export const conversationService = new ConversationService();