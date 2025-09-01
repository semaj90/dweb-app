import type { RequestHandler } from './$types.js';

// Database Orchestrator Events API
// Real-time event monitoring and WebSocket integration

databaseOrchestrator // alias
import { EventEmitter } from "events";
import { URL } from "url";

// GET /api/database-orchestrator/events - Get recent events
export const GET: RequestHandler = async ({ url }) => {
  try {
    const limit = parseInt(url.searchParams.get('limit') || '50');
    const eventType = url.searchParams.get('type');
    const since = url.searchParams.get('since');

    // Create event listener to capture recent events
    const events: any[] = [];
    const eventCollector = (eventData: any) => {
      events.push({
        ...eventData,
        timestamp: new Date().toISOString(),
      });
    };

    // Subscribe to various event types
    const eventTypes = [
      'database:operation_completed',
      'database:new_cases',
      'database:new_evidence',
      'database:new_documents',
      'condition:triggered',
      'health:check',
      'orchestrator:started',
      'orchestrator:stopped',
    ];

    // Add temporary listeners
    eventTypes.forEach((type) => {
      databaseOrchestrator.on(type, eventCollector);
    });

    // Wait briefly to collect any immediate events
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Remove temporary listeners
    eventTypes.forEach((type) => {
      databaseOrchestrator.off(type, eventCollector);
    });

    // Filter events if needed
    let filteredEvents = events;
    if (eventType) {
      filteredEvents = events.filter((e: any) => e.type === eventType);
    }
    if (since) {
      const sinceDate = new Date(since);
      filteredEvents = filteredEvents.filter((e: any) => new Date(e.timestamp) > sinceDate);
    }

    // Limit results
    filteredEvents = filteredEvents.slice(0, limit);

    return json({
      success: true,
      events: filteredEvents,
      count: filteredEvents.length,
      total_available: events.length,
      filters: {
        type: eventType,
        since,
        limit,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};

// POST /api/database-orchestrator/events - Trigger custom events
export const POST: RequestHandler = async ({ request }) => {
  try {
    const { eventType, data, metadata } = await request.json();

    if (!eventType) {
      return json(
        {
          success: false,
          error: 'Event type is required',
        },
        { status: 400 }
      );
    }

    const eventData = {
      type: eventType,
      data: data || {},
      metadata: metadata || {},
      timestamp: new Date().toISOString(),
      source: 'api',
    };

    // Emit the custom event
    databaseOrchestrator.emit(eventType, eventData);

    return json({
      success: true,
      message: 'Event triggered successfully',
      event: eventData,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
};
