// CRUD Test Page Server Logic - SSR & Database Integration
import type { PageServerLoad, Actions } from './$types';
import { fail, redirect } from '@sveltejs/kit';
import { z } from 'zod';

// Database imports
import { db } from '$lib/server/db/index';
import { cases } from '$lib/server/db/schema-postgres';
import { eq, desc, count } from '$lib/server/db/index';

// Validation schema
const caseSchema = z.object({
  title: z.string().min(1, 'Title is required').max(255, 'Title too long'),
  description: z.string().optional(),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).default('medium'),
  status: z.enum(['open', 'active', 'under_review', 'closed', 'archived']).default('open'),
  category: z.string().min(1, 'Category is required').max(100, 'Category too long'),
});

// Database health check
async function checkDatabaseHealth() {
  try {
    const startTime = Date.now();
    
    // Test basic connectivity
    await db.select().from(cases).limit(1);
    const responseTime = Date.now() - startTime;
    
    // Get case count for additional info
    const [{ totalCases }] = await db.select({ totalCases: count() }).from(cases);
    
    return {
      connected: true,
      responseTime,
      totalCases,
      error: null
    };
  } catch (error: any) {
    console.error('Database health check failed:', error);
    return {
      connected: false,
      responseTime: null,
      totalCases: 0,
      error: error.message || 'Database connection failed'
    };
  }
}

// Load page data with SSR
export const load: PageServerLoad = async ({ url, locals }) => {
  try {
    console.log('🔄 Loading CRUD test page with SSR...');

    // Get database health first
    const databaseHealth = await checkDatabaseHealth();
    
    let casesData = [];
    
    // Only fetch cases if database is healthy
    if (databaseHealth.connected) {
      try {
        casesData = await db
          .select()
          .from(cases)
          .orderBy(desc(cases.created_at))
          .limit(20);
        
        console.log(`✅ Loaded ${casesData.length} cases from database`);
      } catch (error) {
        console.error('Failed to load cases:', error);
        // Continue with empty array if case loading fails
      }
    } else {
      console.warn('⚠️ Database not healthy, using empty cases array');
    }

    // System health info
    const health = {
      database: databaseHealth,
      server: {
        connected: true,
        timestamp: new Date().toISOString(),
        url: url.pathname,
        userAgent: locals?.userAgent || 'unknown'
      }
    };

    return {
      cases: casesData,
      health,
      meta: {
        title: 'CRUD Test - Legal AI Platform',
        description: 'Testing SSR, UI Components & Database Operations',
        loadTime: new Date().toISOString()
      }
    };

  } catch (error: any) {
    console.error('❌ Error loading CRUD test page:', error);
    
    // Return fallback data for graceful degradation
    return {
      cases: [],
      health: {
        database: {
          connected: false,
          error: error.message || 'Unknown error',
          responseTime: null,
          totalCases: 0
        },
        server: {
          connected: true,
          timestamp: new Date().toISOString(),
          url: url.pathname
        }
      },
      meta: {
        title: 'CRUD Test - Error State',
        description: 'Database connection failed',
        loadTime: new Date().toISOString()
      }
    };
  }
};

// Form actions for CRUD operations
export const actions: Actions = {
  // Create or Update Case
  createCase: async ({ request, locals }) => {
    try {
      const data = await request.formData();
      const action = data.get('_action');
      const id = data.get('id');
      
      // Extract form data
      const formData = {
        title: data.get('title'),
        description: data.get('description'),
        priority: data.get('priority'),
        status: data.get('status'),
        category: data.get('category'),
      };

      console.log('📝 Processing case form:', { action, id, formData });

      // Validate the form data
      const validation = caseSchema.safeParse(formData);
      if (!validation.success) {
        console.error('❌ Validation failed:', validation.error.flatten());
        return fail(400, {
          error: 'Validation failed',
          issues: validation.error.flatten().fieldErrors,
          formData
        });
      }

      const validData = validation.data;

      // Check database health before operations
      const health = await checkDatabaseHealth();
      if (!health.connected) {
        console.error('❌ Database not available for case operations');
        return fail(503, {
          error: 'Database not available',
          message: health.error,
          formData
        });
      }

      if (action === 'update' && id) {
        // Update existing case
        console.log('🔄 Updating case:', id);
        
        const [updatedCase] = await db
          .update(cases)
          .set({
            ...validData,
            updated_at: new Date()
          })
          .where(eq(cases.id, id as string))
          .returning();

        if (!updatedCase) {
          return fail(404, {
            error: 'Case not found',
            formData
          });
        }

        console.log('✅ Case updated successfully:', updatedCase.id);
        return {
          success: true,
          message: 'Case updated successfully',
          case: updatedCase
        };

      } else {
        // Create new case
        console.log('➕ Creating new case');
        
        const [newCase] = await db
          .insert(cases)
          .values({
            ...validData,
            // Add required fields for the schema
            case_number: `CASE-${Date.now()}`,
            // Use assigned_attorney field from actual schema
            assigned_attorney: locals?.session?.user?.id || locals?.user?.id || null,
            created_at: new Date(),
            updated_at: new Date()
          })
          .returning();

        console.log('✅ Case created successfully:', newCase.id);
        return {
          success: true,
          message: 'Case created successfully',
          case: newCase
        };
      }

    } catch (error: any) {
      console.error('❌ Error processing case form:', error);
      
      return fail(500, {
        error: 'Internal server error',
        message: error.message || 'Failed to process case',
        formData: Object.fromEntries(await request.formData())
      });
    }
  },

  // Delete Case
  deleteCase: async ({ url }) => {
    try {
      const caseId = url.searchParams.get('id');
      
      if (!caseId) {
        return fail(400, { error: 'Case ID is required' });
      }

      console.log('🗑️ Deleting case:', caseId);

      // Check database health
      const health = await checkDatabaseHealth();
      if (!health.connected) {
        return fail(503, {
          error: 'Database not available',
          message: health.error
        });
      }

      // Delete the case
      const [deletedCase] = await db
        .delete(cases)
        .where(eq(cases.id, caseId))
        .returning();

      if (!deletedCase) {
        return fail(404, { error: 'Case not found' });
      }

      console.log('✅ Case deleted successfully:', deletedCase.id);
      return {
        success: true,
        message: 'Case deleted successfully',
        deletedId: deletedCase.id
      };

    } catch (error: any) {
      console.error('❌ Error deleting case:', error);
      return fail(500, {
        error: 'Internal server error',
        message: error.message || 'Failed to delete case'
      });
    }
  }
};