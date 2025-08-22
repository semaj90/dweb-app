// GET /api/user/activity - Get user activity
import type { RequestHandler } from './$types';
import { json } from '@sveltejs/kit';
import { db } from '$lib/yorha/db';
import { userActivity } from '$lib/yorha/db/schema';
import { eq, desc, and, gte, lte } from 'drizzle-orm';
import { AuthService } from '$lib/yorha/services/auth.service';
import { VectorService } from '$lib/yorha/services/vector.service';

export const GET: RequestHandler = async ({ cookies, url }) => {
  try {
    const sessionToken = cookies.get('yorha_session');
    
    if (!sessionToken) {
      return json({
        success: false,
        error: 'Authentication required'
      }, { status: 401 });
    }
    
    const authService = new AuthService();
    const sessionData = await authService.validateSession(sessionToken);
    
    if (!sessionData) {
      return json({
        success: false,
        error: 'Invalid session'
      }, { status: 401 });
    }
    
    // Parse query parameters
    const limit = parseInt(url.searchParams.get('limit') || '50');
    const offset = parseInt(url.searchParams.get('offset') || '0');
    const activityType = url.searchParams.get('type');
    const startDate = url.searchParams.get('startDate');
    const endDate = url.searchParams.get('endDate');
    const search = url.searchParams.get('search');
    
    // Build query conditions
    const conditions = [eq(userActivity.userId, sessionData.unit.id)];
    
    if (activityType) {
      conditions.push(eq(userActivity.activityType, activityType as any));
    }
    
    if (startDate) {
      conditions.push(gte(userActivity.createdAt, new Date(startDate)));
    }
    
    if (endDate) {
      conditions.push(lte(userActivity.createdAt, new Date(endDate)));
    }
    
    // If search query provided, use vector search
    if (search) {
      const vectorService = new VectorService();
      const searchResults = await vectorService.searchActivities(
        search, 
        sessionData.unit.id,
        limit
      );
      
      // Get activity IDs from vector search
      const activityIds = searchResults.map((r: unknown) => r.payload.activityId);
      
      // Fetch full activity records
      const activities = await db.query.userActivity.findMany({
        where: and(
          eq(userActivity.userId, sessionData.unit.id),
          // Add ID filter from vector search
          activityIds.length > 0 ? 
            sql`${userActivity.id} = ANY(${activityIds})` : 
            undefined
        ),
        orderBy: [desc(userActivity.createdAt)],
        limit,
        offset
      });
      
      return json({
        success: true,
        data: {
          activities,
          total: activities.length,
          limit,
          offset,
          searchResults: searchResults.map((r: unknown) => ({
            score: r.score,
            ...r.payload
          }))
        }
      });
    }
    
    // Regular query without search
    const activities = await db.query.userActivity.findMany({
      where: and(...conditions),
      orderBy: [desc(userActivity.createdAt)],
      limit,
      offset
    });
    
    // Get total count
    const [countResult] = await db
      .select({ count: sql`count(*)` })
      .from(userActivity)
      .where(and(...conditions));
    
    const total = Number(countResult.count);
    
    // Group activities by date for better visualization
    const groupedActivities = activities.reduce((acc: unknown, activity) => {
      const date = new Date(activity.createdAt).toLocaleDateString();
      if (!acc[date]) {
        acc[date] = [];
      }
      acc[date].push(activity);
      return acc;
    }, {});
    
    // Calculate activity statistics
    const stats = {
      totalActivities: total,
      activitiesThisWeek: activities.filter(a => {
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        return new Date(a.createdAt) >= weekAgo;
      }).length,
      mostCommonType: activities.reduce((acc: unknown, activity) => {
        acc[activity.activityType] = (acc[activity.activityType] || 0) + 1;
        return acc;
      }, {}),
      lastActivity: activities[0]?.createdAt
    };
    
    return json({
      success: true,
      data: {
        activities,
        groupedActivities,
        stats,
        pagination: {
          total,
          limit,
          offset,
          hasMore: offset + limit < total
        }
      }
    });
  } catch (error: unknown) {
    console.error('Get activity error:', error);
    return json({
      success: false,
      error: 'Failed to fetch activity'
    }, { status: 500 });
  }
};