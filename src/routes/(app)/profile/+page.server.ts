// Profile Page Server Load
import type { PageServerLoad } from './$types';
import { redirect } from '@sveltejs/kit';
import { db } from '$lib/yorha/db';
import { userActivity, userAchievements, userEquipment } from '$lib/yorha/db/schema';
import { eq, desc } from 'drizzle-orm';

export const load: PageServerLoad = async ({ locals }) => {
  // Redirect if not authenticated
  if (!locals.user) {
    throw redirect(302, '/login');
  }
  
  // Fetch user activity
  const activities = await db.query.userActivity.findMany({
    where: eq(userActivity.userId, locals.user.id),
    orderBy: [desc(userActivity.createdAt)],
    limit: 20
  });
  
  // Fetch user achievements
  const achievements = await db.query.userAchievements.findMany({
    where: eq(userAchievements.userId, locals.user.id),
    with: {
      achievement: true
    }
  });
  
  // Fetch user equipment
  const equipment = await db.query.userEquipment.findMany({
    where: eq(userEquipment.userId, locals.user.id),
    with: {
      equipment: true
    }
  });
  
  return {
    user: locals.user,
    activities,
    achievements,
    equipment
  };
};