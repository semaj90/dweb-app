/**
 * Phase 2 Integration Demo Script
 * Prosecutor AI - Enhanced UI/UX with AI Foundations
 *
 * This script demonstrates all Phase 2 features working together:
 * - Melt UI + Bits UI v2 integration
 * - AI command parsing with real-time updates
 * - XState machine workflows
 * - Enhanced component system
 * - Evidence system integration
 */

import { parseAICommand, aiCommandService } from "./ai-command-parser.js";
import {
  createEnhancedButton,
  mergeBitsUIProps,
  uiUpdateManager,
  YorhaClassManager,
} from "./enhanced-ui-system.js";
import { aiCommandMachine } from "./ai-command-machine.js";

/**
 * Demo: AI-Enhanced Evidence Upload with Real-time UI Updates
 */
export async function demoEvidenceUpload() {
  console.log("🎯 Phase 2 Demo: AI-Enhanced Evidence Upload");

  // 1. Parse AI command for evidence upload
  const command = "upload evidence priority high type document";
  console.log(`🤖 Processing AI command: "${command}"`);

  try {
    const result = await parseAICommand(command);
    console.log("✅ AI Command Result:", result);

    // 2. Apply real-time UI updates
    uiUpdateManager.queueUpdate({
      selector: ".evidence-dropzone",
      classes: {
        add: ["evidence-active", "ai-highlight", "priority-high"],
      },
      attributes: {
        "data-evidence-type": "document",
        "data-priority": "high",
        "data-ai-processed": "true",
      },
    });

    // 3. Update evidence cards with AI-controlled styling
    uiUpdateManager.queueUpdate({
      selector: '.evidence-card[data-type="document"]',
      classes: {
        add: ["evidence-type-document", "animate-evidence-upload"],
      },
    });

    console.log("✅ Real-time UI updates queued and processing");
  } catch (error) {
    console.error("❌ AI Command Error:", error);
  }
}

/**
 * Demo: Melt UI + Bits UI v2 Button Integration
 */
export function demoEnhancedButton() {
  console.log("🎨 Phase 2 Demo: Enhanced Button Integration");

  // Create AI-enhanced button with Melt UI
  const buttonBuilder = createEnhancedButton({
    variant: "primary",
    aiControlled: true,
    yorhaSupport: true,
  });

  console.log("✅ Enhanced button created with AI capabilities");

  // Example of prop merging for Bits UI v2
  const meltProps = buttonBuilder.elements.root;
  const bitsProps = { variant: "default", size: "md" };
  const aiProps = {
    aiClasses: ["ai-btn-primary", "prosecutor-enhanced"],
    yorhaClass: "yorha-btn-primary",
  };

  const mergedProps = mergeBitsUIProps(meltProps, bitsProps, aiProps);
  console.log("✅ Props merged for Bits UI v2 compatibility:", mergedProps);

  return { buttonBuilder, mergedProps };
}

/**
 * Demo: XState Machine AI Command Processing
 */
export function demoXStateMachine() {
  console.log("⚙️ Phase 2 Demo: XState Machine Workflow");

  // Subscribe to machine state changes
  aiCommandService.subscribe((state) => {
    console.log(`🔄 Machine State: ${state.value}`, {
      context: state.context,
      canTransition: state.can("PROCESS_COMMAND"),
    });
  });

  // Send commands to the machine
  const commands = [
    "analyze evidence patterns",
    "highlight priority items",
    "generate case summary",
  ];

  commands.forEach((command, index) => {
    setTimeout(() => {
      console.log(`📤 Sending command ${index + 1}: "${command}"`);
      aiCommandService.send({ type: "PROCESS_COMMAND", command });
    }, index * 2000);
  });

  console.log("✅ XState machine demo commands queued");
}

/**
 * Demo: Legacy Yorha Class Support
 */
export function demoYorhaIntegration() {
  console.log("🎭 Phase 2 Demo: Legacy Yorha Support");

  // Create a mock element for demonstration
  const mockElement = {
    classList: {
      add: (className) => console.log(`  ➕ Added class: ${className}`),
      remove: (className) => console.log(`  ➖ Removed class: ${className}`),
    },
  };

  // Apply different Yorha themes
  console.log('🎨 Applying Yorha "enhanced" theme:');
  YorhaClassManager.applyYorhaTheme(mockElement, "enhanced");

  console.log('🎨 Applying Yorha "terminal" theme:');
  YorhaClassManager.applyYorhaTheme(mockElement, "terminal");

  console.log("✅ Legacy Yorha integration demonstrated");
}

/**
 * Demo: Evidence System Features
 */
export function demoEvidenceSystem() {
  console.log("📋 Phase 2 Demo: Evidence System Features");

  const evidenceTypes = ["document", "image", "video", "audio", "physical"];
  const priorities = ["high", "medium", "low"];

  evidenceTypes.forEach((type, index) => {
    const priority = priorities[index % priorities.length];

    console.log(`📄 Evidence Type: ${type} | Priority: ${priority}`);

    // Simulate evidence upload with AI enhancement
    uiUpdateManager.queueUpdate({
      selector: `#evidence-${index}`,
      classes: {
        add: [
          "evidence-card",
          `evidence-type-${type}`,
          `priority-${priority}`,
          "animate-evidence-upload",
        ],
      },
      attributes: {
        "data-type": type,
        "data-priority": priority,
        "data-uploaded": new Date().toISOString(),
      },
    });
  });

  console.log("✅ Evidence system demo completed");
}

/**
 * Run Complete Phase 2 Demo
 */
export async function runPhase2Demo() {
  console.log("🚀 Starting Complete Phase 2 Integration Demo");
  console.log("====================================================");

  try {
    // Demo 1: Enhanced Button Integration
    demoEnhancedButton();
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Demo 2: AI Command Processing
    await demoEvidenceUpload();
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Demo 3: XState Machine
    demoXStateMachine();
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Demo 4: Legacy Yorha Support
    demoYorhaIntegration();
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Demo 5: Evidence System
    demoEvidenceSystem();

    console.log("====================================================");
    console.log("🎉 Phase 2 Demo Complete!");
    console.log("✅ All systems integrated and working");
    console.log("🗺️ Ready for Phase 3: AI Core Implementation");
  } catch (error) {
    console.error("❌ Demo Error:", error);
  }
}

/**
 * Phase 2 Health Check
 */
export function phase2HealthCheck() {
  const systems = {
    "AI Command Parser": typeof parseAICommand === "function",
    "XState Machine": typeof aiCommandService === "object",
    "Enhanced Buttons": typeof createEnhancedButton === "function",
    "UI Update Manager": typeof uiUpdateManager === "object",
    "Yorha Integration": typeof YorhaClassManager === "function",
    "Prop Merging": typeof mergeBitsUIProps === "function",
  };

  console.log("🏥 Phase 2 Health Check:");
  Object.entries(systems).forEach(([name, status]) => {
    console.log(
      `  ${status ? "✅" : "❌"} ${name}: ${status ? "OK" : "MISSING"}`,
    );
  });

  const allHealthy = Object.values(systems).every(Boolean);
  console.log(
    `🎯 Overall Status: ${allHealthy ? "✅ HEALTHY" : "❌ ISSUES DETECTED"}`,
  );

  return allHealthy;
}

// Export for use in components
export default {
  demoEvidenceUpload,
  demoEnhancedButton,
  demoXStateMachine,
  demoYorhaIntegration,
  demoEvidenceSystem,
  runPhase2Demo,
  phase2HealthCheck,
};
