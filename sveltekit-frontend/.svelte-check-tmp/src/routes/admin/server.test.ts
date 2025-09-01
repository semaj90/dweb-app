/* eslint-disable @typescript-eslint/no-explicit-any */

/*
 Provide minimal ambient declarations so the test file doesn't require
 having the 'vitest' type package available to TypeScript while still
 allowing the test runner to provide the real globals at runtime.
*/
// Use runtime globals from globalThis without declaring names that may collide
const gvi: any = (globalThis as any).vi;
const gdescribe: any = (globalThis as any).describe;
const git: any = (globalThis as any).it;
const gexpect: any = (globalThis as any).expect;
const gbeforeEach: any = (globalThis as any).beforeEach;

// We intentionally avoid declaring an ambient module for './+server'
// because TypeScript does not accept relative names in ambient module
// declarations in this context; tests will import the runtime handler using
// a runtime dynamic import and we suppress TS errors at the import site.

/* Minimal UnitRecord type used by tests */
type UnitRecord = {
  id: string;
  unitId: string;
  email?: string;
  name?: string;
  unitType?: string;
  level?: number;
  xp?: number;
  rank?: string;
  bio?: string | null;
  avatarUrl?: string | null;
  missionsCompleted?: number;
  combatRating?: number;
  hoursActive?: number;
  achievementsUnlocked?: number;
  emailVerified?: boolean;
  twoFactorEnabled?: boolean;
  settings?: Record<string, any>;
  embeddingUpdatedAt?: Date | null;
  embeddingStatus?: string | null;
  createdAt?: Date;
  updatedAt?: Date | null;
};

/* In-memory mock DB used by the test runtime */
const mockDb: any = {
  _units: {} as Record<string, UnitRecord & Record<string, any>>,
  query: {
    units: {
      async findFirst(_where?: any) {
        // we don't parse 'where' expression; tests will only call when needed
        // return the first unit if exists
        return Object.values(mockDb._units)[0] ?? null;
      }
    },
    userActivity: {
      update(_table: any) {
        const chain: any = {
          _set: undefined,
          set(data: any) {
            this._set = data;
            return this;
          },
          where(_: any) {
            return this;
          },
          async returning() {
            // apply this._set to first unit (tests use single unit)
            const unit = Object.values(mockDb._units)[0] as UnitRecord & Record<string, any>;
            if (!unit) return [];
            Object.assign(unit, this._set);
            // ensure updatedAt is Date if present as string
            if (typeof unit.updatedAt === 'string') unit.updatedAt = new Date(unit.updatedAt);
            return [unit];
          }
        };
        return chain;
      },
      insert(_table: any) {
        return {
          values: async (_v: any) => {
            // noop for activity insert
            return;
          }
        };
      }
    }
  }
};

// spies accessible inside mocks
const vectorGenerateSpy = gvi.fn();

// Mock SvelteKit json helper to return plain object for easy assertions
gvi.mock('@sveltejs/kit', () => {
  return {
    json: (body: any, init?: any) => {
      return { body, status: init?.status ?? 200 };
    }
  };
});

// Mock DB module
gvi.mock('$lib/yorha/db', () => {
  return { db: mockDb };
});

// Mock AuthService to validate sessions and return sessionData containing our in-memory unit
gvi.mock('$lib/yorha/services/auth.service', () => {
  return {
    AuthService: class {
      async validateSession(token: string) {
        if (token !== 'valid-token') return null;
        const unit = Object.values(mockDb._units)[0];
        return {
          session: { id: 'sess-1' },
          unit
        };
      }
    }
  };
});

// Mock VectorService: publishing a job (recorded by spy) and scheduling a background worker
gvi.mock('$lib/yorha/services/vector.service', () => {
  return {
    VectorService: class {
      async generateUserEmbedding(id: string) {
        vectorGenerateSpy(id);
        // schedule background worker to write embeddingUpdatedAt
        setTimeout(() => {
          const u = mockDb._units[id];
          if (u) {
            u.embeddingUpdatedAt = new Date();
            u.embeddingStatus = 'ready';
          }
        }, 10);
        // return quickly to simulate "publish job" (non-blocking worker)
        return Promise.resolve();
      }
    }
  };
});

gdescribe('Profile API integration (mocked DB + worker)', () => {
  gbeforeEach(() => {
    // reset in-memory DB and spies
    for (const k of Object.keys(mockDb._units)) delete mockDb._units[k];
    vectorGenerateSpy.mockReset();

    const unit: UnitRecord = {
      id: 'unit-1',
      unitId: 'u1',
      email: 'test@example.com',
      name: 'Old Name',
      unitType: 'soldier',
      level: 1,
      xp: 0,
      rank: 'Rookie',
      bio: 'Old bio',
      avatarUrl: null,
      missionsCompleted: 0,
      combatRating: 0,
      hoursActive: 0,
      achievementsUnlocked: 0,
      emailVerified: true,
      twoFactorEnabled: false,
      settings: {},
      embeddingUpdatedAt: null,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    mockDb._units[unit.id] = unit;
  });

  git('updates profile via PUT and returns embedding metadata via GET', async () => {
    // import handlers after mocks are set up
    // TypeScript can't find type declarations for the generated +server module in this test environment,
    // so ignore type checking for this dynamic import and assert the shape afterwards.
    // @ts-ignore
    const mod = await import('./+server');
    const { PUT, GET } = mod as { PUT: any; GET: any };

    const cookies = {
      get: (k: string) => (k === 'yorha_session' ? 'valid-token' : undefined)
    };

    // prepare a request object for PUT (update name)
    const request = {
      json: async () => ({ name: 'New Name' })
    };

    // call PUT handler
    const putResult = await PUT({ request, cookies });
    // verify PUT responded success
    gexpect(putResult).toBeDefined();
    gexpect(putResult.body?.success ?? putResult.body?.success).toBe(true);

    // confirm DB unit was updated (updatedAt was set)
    const updatedUnit = mockDb._units['unit-1'];
    gexpect(updatedUnit.name).toBe('New Name');
    gexpect(updatedUnit.updatedAt).toBeInstanceOf(Date);

    // confirm a job was published (VectorService invoked)
    gexpect(vectorGenerateSpy).toHaveBeenCalledTimes(1);
    gexpect(vectorGenerateSpy).toHaveBeenCalledWith('unit-1');

    // wait for the mocked background worker to run and write embeddingUpdatedAt
    await new Promise((r) => setTimeout(r, 20));

    // call GET handler to fetch profile and embedding metadata
    const url = new URL('http://localhost/');
    const getCookies = { get: (k: string) => (k === 'yorha_session' ? 'valid-token' : undefined) };
    const getResult = await GET({ cookies: getCookies, url });
    gexpect(getResult).toBeDefined();
    const body = getResult.body ?? getResult.body;
    gexpect(body.success).toBe(true);
    const embedding = body.data.embedding;
    gexpect(embedding).toBeDefined();
    gexpect(embedding.lastUpdatedAt).not.toBeNull();
    // embedding.lastUpdatedAt should be a Date (mock returned Date object)
    gexpect(embedding.lastUpdatedAt).toBeInstanceOf(Date);
    gexpect(mockDb._units['unit-1'].embeddingStatus).toBe('ready');
  });
});

export {};