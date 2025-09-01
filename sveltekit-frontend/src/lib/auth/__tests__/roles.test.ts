import { describe, it, expect } from 'vitest';
import { AccessControl, ROLES, ROLE_HIERARCHY, type UserRole } from '../../auth/roles';

describe('roles & access control', () => {
  it('each role has non-empty permissions', () => {
    for (const role of Object.keys(ROLES) as UserRole[]) {
      expect(ROLES[role].permissions.length).toBeGreaterThan(0);
    }
  });

  it('hierarchy ordering is descending in levels', () => {
    for (let i = 0; i < ROLE_HIERARCHY.length - 1; i++) {
      const a = ROLES[ROLE_HIERARCHY[i]].hierarchyLevel;
      const b = ROLES[ROLE_HIERARCHY[i+1]].hierarchyLevel;
      expect(a).toBeGreaterThanOrEqual(b);
    }
  });

  it('admin has all permissions', () => {
    const adminPerms = new Set(ROLES.admin.permissions);
    for (const role of Object.keys(ROLES) as UserRole[]) {
      for (const p of ROLES[role].permissions) {
        if (!adminPerms.has(p)) throw new Error(`Admin missing permission ${p}`);
      }
    }
  });

  it('higher authority check works', () => {
    expect(AccessControl.hasHigherAuthority('admin','prosecutor')).toBe(true);
    expect(AccessControl.hasHigherAuthority('prosecutor','admin')).toBe(false);
  });

  it('restricted delete permissions only for allowed roles', () => {
    expect(AccessControl.hasPermission('prosecutor','delete_case' as any)).toBe(false);
    expect(AccessControl.hasPermission('admin','delete_case' as any)).toBe(true);
  });
});
