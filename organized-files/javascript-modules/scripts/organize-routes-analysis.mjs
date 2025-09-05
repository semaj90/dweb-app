#!/usr/bin/env node

/**
 * ROUTE ORGANIZATION ANALYSIS
 * Categorizes and analyzes route usage across the SvelteKit frontend
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const routesDir = path.resolve(__dirname, '../sveltekit-frontend/src/routes');
const srcDir = path.resolve(__dirname, '../sveltekit-frontend/src');

// Route categorization patterns
const ROUTE_CATEGORIES = {
  core: {
    name: 'Core Application',
    patterns: [/^\/$/, /^\/cases/, /^\/evidence/, /^\/documents/, /^\/legal/, /^\/search/],
    priority: 'high'
  },
  auth: {
    name: 'Authentication',
    patterns: [/^\/login/, /^\/register/, /^\/logout/, /^\/profile/],
    priority: 'high'
  },
  admin: {
    name: 'Administrative',
    patterns: [/^\/admin/],
    priority: 'medium'
  },
  api: {
    name: 'API Endpoints',
    patterns: [/^\/api/],
    priority: 'high'
  },
  ai: {
    name: 'AI Features',
    patterns: [/^\/ai/, /^\/rag/, /^\/context7/, /^\/semantic/],
    priority: 'high'
  },
  demos: {
    name: 'Demos & Testing',
    patterns: [/demo/, /test/, /showcase/, /^\/dev/, /^\/perf/],
    priority: 'low'
  },
  utilities: {
    name: 'Utility Pages',
    patterns: [/^\/help/, /^\/settings/, /^\/dashboard/, /^\/upload/, /^\/export/, /^\/import/],
    priority: 'medium'
  },
  legacy: {
    name: 'Legacy/Experimental',
    patterns: [/original/, /old/, /backup/, /legacy/, /disabled/],
    priority: 'low'
  }
};

async function scanRoutesDirectory() {
  const routes = [];
  
  async function scanDir(dir, basePath = '') {
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        if (entry.isDirectory() && !entry.name.startsWith('.')) {
          const fullPath = path.join(dir, entry.name);
          const routePath = basePath + '/' + entry.name;
          
          // Check for page files in this directory
          const hasPageFile = await checkForPageFile(fullPath);
          const hasServerFile = await checkForServerFile(fullPath);
          const hasLayoutFile = await checkForLayoutFile(fullPath);
          
          routes.push({
            path: routePath,
            directory: fullPath,
            hasPageFile,
            hasServerFile,
            hasLayoutFile,
            isParametric: entry.name.includes('['),
            isGrouped: entry.name.startsWith('('),
            category: categorizeRoute(routePath)
          });
          
          // Recursively scan subdirectories
          await scanDir(fullPath, routePath);
        }
      }
    } catch (error) {
      console.warn(`⚠️ Could not scan ${dir}: ${error.message}`);
    }
  }
  
  await scanDir(routesDir);
  return routes;
}

async function checkForPageFile(dir) {
  try {
    await fs.access(path.join(dir, '+page.svelte'));
    return true;
  } catch {
    return false;
  }
}

async function checkForServerFile(dir) {
  try {
    const serverFiles = ['+page.server.ts', '+page.server.js', '+layout.server.ts', '+layout.server.js'];
    for (const file of serverFiles) {
      try {
        await fs.access(path.join(dir, file));
        return true;
      } catch {}
    }
    return false;
  } catch {
    return false;
  }
}

async function checkForLayoutFile(dir) {
  try {
    const layoutFiles = ['+layout.svelte', '+layout.ts', '+layout.js'];
    for (const file of layoutFiles) {
      try {
        await fs.access(path.join(dir, file));
        return true;
      } catch {}
    }
    return false;
  } catch {
    return false;
  }
}

function categorizeRoute(routePath) {
  for (const [key, category] of Object.entries(ROUTE_CATEGORIES)) {
    if (category.patterns.some(pattern => pattern.test(routePath))) {
      return { key, ...category };
    }
  }
  return { key: 'uncategorized', name: 'Uncategorized', priority: 'low' };
}

async function analyzeRouteUsage(routes) {
  const usageAnalysis = {
    totalRoutes: routes.length,
    functionalRoutes: routes.filter(r => r.hasPageFile).length,
    apiRoutes: routes.filter(r => r.hasServerFile).length,
    layoutRoutes: routes.filter(r => r.hasLayoutFile).length,
    emptyDirectories: routes.filter(r => !r.hasPageFile && !r.hasServerFile && !r.hasLayoutFile).length
  };

  // Check for navigation references
  const navigationRefs = await findNavigationReferences(routes);
  
  // Analyze by category
  const categoryStats = {};
  routes.forEach(route => {
    const catKey = route.category.key;
    if (!categoryStats[catKey]) {
      categoryStats[catKey] = {
        name: route.category.name,
        priority: route.category.priority,
        total: 0,
        functional: 0,
        empty: 0,
        routes: []
      };
    }
    categoryStats[catKey].total++;
    if (route.hasPageFile) categoryStats[catKey].functional++;
    if (!route.hasPageFile && !route.hasServerFile && !route.hasLayoutFile) categoryStats[catKey].empty++;
    categoryStats[catKey].routes.push(route);
  });

  return { usageAnalysis, categoryStats, navigationRefs };
}

async function findNavigationReferences(routes) {
  const navigationRefs = {};
  const searchPatterns = routes.map(r => r.path);
  
  try {
    // Search in common navigation files
    const navFiles = [
      'src/lib/components/Navigation.svelte',
      'src/lib/components/Header.svelte',
      'src/lib/components/Sidebar.svelte',
      'src/lib/components/yorha/YoRHaNavigation.svelte',
      'src/routes/+layout.svelte'
    ];
    
    for (const file of navFiles) {
      try {
        const fullPath = path.resolve(__dirname, '../sveltekit-frontend', file);
        const content = await fs.readFile(fullPath, 'utf-8');
        
        searchPatterns.forEach(routePath => {
          if (content.includes(`href="${routePath}`) || content.includes(`'${routePath}'`) || content.includes(`"${routePath}"`)) {
            if (!navigationRefs[routePath]) navigationRefs[routePath] = [];
            navigationRefs[routePath].push(file);
          }
        });
      } catch {
        // File doesn't exist
      }
    }
  } catch (error) {
    console.warn(`⚠️ Could not analyze navigation references: ${error.message}`);
  }
  
  return navigationRefs;
}

async function generateCleanupRecommendations(routes, analysisResults) {
  const recommendations = {
    canRemove: [],
    shouldKeep: [],
    needsReview: [],
    consolidate: []
  };

  routes.forEach(route => {
    const hasNavRef = analysisResults.navigationRefs[route.path];
    const isEmpty = !route.hasPageFile && !route.hasServerFile && !route.hasLayoutFile;
    const isDemo = route.category.key === 'demos';
    const isLegacy = route.category.key === 'legacy';
    
    if (isEmpty && !hasNavRef && (isDemo || isLegacy)) {
      recommendations.canRemove.push({
        path: route.path,
        reason: 'Empty directory, no navigation references, low priority category'
      });
    } else if (route.category.priority === 'high' || hasNavRef) {
      recommendations.shouldKeep.push({
        path: route.path,
        reason: 'High priority or has navigation references'
      });
    } else if (isEmpty) {
      recommendations.needsReview.push({
        path: route.path,
        reason: 'Empty directory but might be planned feature'
      });
    }
  });

  // Find consolidation opportunities
  const duplicatePatterns = {};
  routes.forEach(route => {
    const baseName = route.path.split('/').pop();
    if (baseName.includes('demo') || baseName.includes('test')) {
      if (!duplicatePatterns[baseName]) duplicatePatterns[baseName] = [];
      duplicatePatterns[baseName].push(route.path);
    }
  });

  Object.entries(duplicatePatterns).forEach(([pattern, paths]) => {
    if (paths.length > 1) {
      recommendations.consolidate.push({
        pattern,
        paths,
        reason: `Multiple ${pattern} routes could be consolidated`
      });
    }
  });

  return recommendations;
}

async function main() {
  console.log('🎯 ROUTE ORGANIZATION ANALYSIS');
  console.log('📋 Scanning SvelteKit routes directory...');
  
  const startTime = Date.now();
  const routes = await scanRoutesDirectory();
  const analysisResults = await analyzeRouteUsage(routes);
  const recommendations = await generateCleanupRecommendations(routes, analysisResults);
  
  const duration = ((Date.now() - startTime) / 1000).toFixed(1);

  // Console output
  console.log('\n📊 ROUTE ANALYSIS RESULTS');
  console.log('════════════════════════════════════════');
  console.log(`📁 Total routes discovered: ${analysisResults.usageAnalysis.totalRoutes}`);
  console.log(`✅ Functional routes (have +page.svelte): ${analysisResults.usageAnalysis.functionalRoutes}`);
  console.log(`🔌 API routes (have server files): ${analysisResults.usageAnalysis.apiRoutes}`);
  console.log(`📐 Layout routes (have +layout files): ${analysisResults.usageAnalysis.layoutRoutes}`);
  console.log(`📂 Empty directories: ${analysisResults.usageAnalysis.emptyDirectories}`);
  console.log(`⏱️ Analysis time: ${duration}s`);

  console.log('\n📂 ROUTES BY CATEGORY:');
  Object.entries(analysisResults.categoryStats)
    .sort(([,a], [,b]) => {
      const priorities = { high: 3, medium: 2, low: 1 };
      return (priorities[b.priority] || 0) - (priorities[a.priority] || 0);
    })
    .forEach(([key, stats]) => {
      const priority = stats.priority === 'high' ? '🔥' : stats.priority === 'medium' ? '⚡' : '📝';
      console.log(`  ${priority} ${stats.name}: ${stats.functional}/${stats.total} functional (${stats.empty} empty)`);
    });

  console.log('\n🧹 CLEANUP RECOMMENDATIONS:');
  console.log(`🗑️ Can remove: ${recommendations.canRemove.length} routes`);
  console.log(`✅ Should keep: ${recommendations.shouldKeep.length} routes`);
  console.log(`🔍 Need review: ${recommendations.needsReview.length} routes`);
  console.log(`🔀 Can consolidate: ${recommendations.consolidate.length} patterns`);

  if (recommendations.canRemove.length > 0) {
    console.log('\n🗑️ SAFE TO REMOVE:');
    recommendations.canRemove.slice(0, 10).forEach(item => {
      console.log(`   • ${item.path} - ${item.reason}`);
    });
    if (recommendations.canRemove.length > 10) {
      console.log(`   ... and ${recommendations.canRemove.length - 10} more`);
    }
  }

  // Generate detailed report
  const reportPath = path.resolve(__dirname, '../archives/component-backups/ROUTE_ORGANIZATION_ANALYSIS.md');
  await generateDetailedReport(routes, analysisResults, recommendations, reportPath);
  
  console.log(`\n📋 Detailed report saved: ${reportPath}`);
  console.log('\n✅ Route organization analysis complete!');
}

async function generateDetailedReport(routes, analysisResults, recommendations, reportPath) {
  let report = `# Route Organization Analysis Report\n`;
  report += `Generated: ${new Date().toISOString()}\n\n`;
  
  report += `## Executive Summary\n`;
  report += `- Total routes: ${analysisResults.usageAnalysis.totalRoutes}\n`;
  report += `- Functional routes: ${analysisResults.usageAnalysis.functionalRoutes}\n`;
  report += `- Empty directories: ${analysisResults.usageAnalysis.emptyDirectories}\n`;
  report += `- Navigation references found: ${Object.keys(analysisResults.navigationRefs).length}\n\n`;

  report += `## Routes by Category\n\n`;
  Object.entries(analysisResults.categoryStats)
    .sort(([,a], [,b]) => {
      const priorities = { high: 3, medium: 2, low: 1 };
      return (priorities[b.priority] || 0) - (priorities[a.priority] || 0);
    })
    .forEach(([key, stats]) => {
      report += `### ${stats.name} (${stats.priority} priority)\n`;
      report += `- Total: ${stats.total} routes\n`;
      report += `- Functional: ${stats.functional} routes\n`;
      report += `- Empty: ${stats.empty} routes\n\n`;
      
      if (stats.routes.length > 0) {
        report += `#### Routes in this category:\n`;
        stats.routes.forEach(route => {
          const status = route.hasPageFile ? '✅' : route.hasServerFile ? '🔌' : '📂';
          report += `- ${status} \`${route.path}\`\n`;
        });
        report += `\n`;
      }
    });

  report += `## Cleanup Recommendations\n\n`;
  
  if (recommendations.canRemove.length > 0) {
    report += `### Safe to Remove (${recommendations.canRemove.length} routes)\n\n`;
    recommendations.canRemove.forEach(item => {
      report += `- \`${item.path}\` - ${item.reason}\n`;
    });
    report += `\n`;
  }

  if (recommendations.needsReview.length > 0) {
    report += `### Need Manual Review (${recommendations.needsReview.length} routes)\n\n`;
    recommendations.needsReview.forEach(item => {
      report += `- \`${item.path}\` - ${item.reason}\n`;
    });
    report += `\n`;
  }

  if (recommendations.consolidate.length > 0) {
    report += `### Consolidation Opportunities (${recommendations.consolidate.length} patterns)\n\n`;
    recommendations.consolidate.forEach(item => {
      report += `**${item.pattern}** - ${item.reason}\n`;
      item.paths.forEach(path => {
        report += `- \`${path}\`\n`;
      });
      report += `\n`;
    });
  }

  report += `## Navigation References\n\n`;
  Object.entries(analysisResults.navigationRefs).forEach(([route, files]) => {
    report += `- \`${route}\` referenced in: ${files.join(', ')}\n`;
  });

  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, report);
}

main().catch(console.error);