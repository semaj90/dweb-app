/**
 * Multi-Agent Orchestrator API Endpoints - Step 9 Integration
 */

import type { RequestHandler } from "./$types";
import { json } from "@sveltejs/kit";
import { multiAgentOrchestrator } from "$lib/services/multi-agent-orchestrator";

// GET /api/orchestrator - Get workflow status or list workflows
export const GET: RequestHandler = async ({ url }) => {
  try {
    const workflowId = url.searchParams.get("workflowId");

    if (workflowId) {
      const workflow = multiAgentOrchestrator.getWorkflowStatus(workflowId);
      if (!workflow) {
        return json(
          {
            success: false,
            error: "Workflow not found",
          },
          { status: 404 }
        );
      }

      return json({
        success: true,
        workflow,
      });
    } else {
      const workflows = await multiAgentOrchestrator.listActiveWorkflows();
      return json({
        success: true,
        workflows,
        count: workflows.workflows.length,
      });
    }
  } catch (error) {
    console.error("Failed to get workflow status:", error);
    return json(
      {
        success: false,
        error: "Failed to get workflow status",
      },
      { status: 500 }
    );
  }
};

// POST /api/orchestrator - Create and execute workflow
export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      action,
      name,
      description,
      capabilities = [],
      query,
      context = {},
      workflowId,
    } = await request.json();

    switch (action) {
      case "create":
        const newWorkflowId = await multiAgentOrchestrator.createWorkflow({
          name,
          description,
          capabilities,
          context: {
            originalQuery: query,
            sessionId: crypto.randomUUID(),
            ...context,
          }
        });

        return json({
          success: true,
          workflowId: newWorkflowId.workflowId,
          message: "Workflow created successfully",
        });

      case "execute":
        if (!workflowId) {
          return json(
            {
              success: false,
              error: "Workflow ID is required for execution",
            },
            { status: 400 }
          );
        }

        const result = await multiAgentOrchestrator.executeWorkflow(
          workflowId,
          {
            originalQuery: query,
            sessionId: crypto.randomUUID(),
            ...context,
          }
        );

        return json({
          success: true,
          workflowId,
          result,
          message: "Workflow executed successfully",
        });

      case "create_and_execute":
        const createAndExecWorkflowId =
          await multiAgentOrchestrator.createWorkflow({
            name: name || "Auto-generated Workflow",
            description: description || "Automatically created and executed workflow",
            capabilities,
            context: {
              originalQuery: query,
              sessionId: crypto.randomUUID(),
              ...context,
            }
          });

        const execResult = await multiAgentOrchestrator.executeWorkflow(
          createAndExecWorkflowId.workflowId,
          {
            originalQuery: query,
            sessionId: crypto.randomUUID(),
            ...context,
          }
        );

        return json({
          success: true,
          workflowId: createAndExecWorkflowId.workflowId,
          result: execResult,
          message: "Workflow created and executed successfully",
        });

      default:
        return json(
          {
            success: false,
            error:
              "Invalid action. Use: create, execute, or create_and_execute",
          },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error("Orchestrator API error:", error);
    return json(
      {
        success: false,
        error: error.message || "Orchestrator operation failed",
      },
      { status: 500 }
    );
  }
};

// DELETE /api/orchestrator - Cancel workflow
export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const workflowId = url.searchParams.get("workflowId");

    if (!workflowId) {
      return json(
        {
          success: false,
          error: "Workflow ID is required",
        },
        { status: 400 }
      );
    }

    await multiAgentOrchestrator.cancelWorkflow(workflowId);

    return json({
      success: true,
      message: "Workflow cancelled successfully",
    });
  } catch (error) {
    console.error("Failed to cancel workflow:", error);
    return json(
      {
        success: false,
        error: "Failed to cancel workflow",
      },
      { status: 500 }
    );
  }
};
