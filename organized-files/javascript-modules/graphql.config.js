/** @type {import('graphql-config').IGraphQLConfig} */
const config = {
  schema: [
    {
      'http://localhost:5173/api/graphql': {
        headers: {
          'Content-Type': 'application/json',
        },
      },
    },
    './src/lib/graphql/schema.ts'
  ],
  documents: [
    './src/**/*.{graphql,js,ts,jsx,tsx,svelte}',
    './sveltekit-frontend/src/**/*.{graphql,js,ts,jsx,tsx,svelte}'
  ],
  extensions: {
    codegen: {
      generates: {
        './src/lib/graphql/generated/types.ts': {
          plugins: [
            'typescript',
            'typescript-operations'
          ],
          config: {
            skipTypename: false,
            withHooks: true,
            withHOC: false,
            withComponent: false,
          },
        },
        './src/lib/graphql/generated/introspection.json': {
          plugins: ['introspection'],
        },
        './sveltekit-frontend/src/lib/graphql/generated/': {
          preset: 'client',
          plugins: [],
          presetConfig: {
            gqlTagName: 'gql',
          }
        }
      },
    },
  },
};

module.exports = config;