import type { RequestHandler } from './$types';

// Database Orchestrator Conditions API
// Manages event loop conditions and real-time triggers

databaseOrchestrator // alias

// GET /api/database-orchestrator/conditions - List all conditions
export const GET: RequestHandler = async () => {
  try {
    const status = databaseOrchestrator.getStatus();
    const conditions = Array.from(databaseOrchestrator.conditions || new Map()).map(
      ([id, condition]) => ({
        id,
        ...condition,
      })
    );

    return json({
      success: true,
      conditions,
      active_count: conditions.filter((c) => c.isActive).length,
      total_count: conditions.length,
      orchestrator_running: status.isRunning,
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

// POST /api/database-orchestrator/conditions - Add new condition
export const POST: RequestHandler = async ({ request }) => {
  try {
    const conditionData = await request.json();

    // Validate required fields
    if (!conditionData.id || !conditionData.type || !conditionData.action) {
      return json(
        {
          success: false,
          error: 'Missing required fields: id, type, action',
        },
        { status: 400 }
      );
    }

    const condition = {
      id: conditionData.id,
      type: conditionData.type,
      condition: conditionData.condition || {},
      action: conditionData.action,
      isActive: conditionData.isActive !== false,
      metadata: conditionData.metadata || {},
    };

    databaseOrchestrator.addCondition(condition);

    return json({
      success: true,
      message: 'Condition added successfully',
      condition,
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

// DELETE /api/database-orchestrator/conditions/:id - Remove condition
export const DELETE: RequestHandler = async ({ params }) => {
  try {
    const { id } = params;

    if (!id) {
      return json(
        {
          success: false,
          error: 'Condition ID is required',
        },
        { status: 400 }
      );
    }

    databaseOrchestrator.removeCondition(id);

    return json({
      success: true,
      message: 'Condition removed successfully',
      conditionId: id,
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
