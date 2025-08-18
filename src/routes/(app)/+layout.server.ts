// Layout Server Load Function
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = async ({ locals }) => {
  return {
    user: locals.user ? {
      id: locals.user.id,
      unitId: locals.user.unitId,
      email: locals.user.email,
      name: locals.user.name,
      unitType: locals.user.unitType,
      level: locals.user.level,
      xp: locals.user.xp,
      rank: locals.user.rank,
      bio: locals.user.bio,
      avatarUrl: locals.user.avatarUrl,
      missionsCompleted: locals.user.missionsCompleted,
      combatRating: locals.user.combatRating,
      hoursActive: locals.user.hoursActive,
      achievementsUnlocked: locals.user.achievementsUnlocked,
      emailVerified: locals.user.emailVerified,
      twoFactorEnabled: locals.user.twoFactorEnabled,
      settings: locals.user.settings,
      createdAt: locals.user.createdAt,
      lastLoginAt: locals.user.lastLoginAt
    } : null
  };
};