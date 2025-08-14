@echo off
echo Installing GraphQL Dependencies for Legal AI System...
echo.

:: Root dependencies
echo Installing root-level GraphQL dependencies...
call npm install graphql@^16.8.1 graphql-yoga@^5.1.1 @pothos/core@^4.2.0 @pothos/plugin-drizzle@^4.2.0

:: Frontend dependencies  
echo.
echo Installing frontend GraphQL dependencies...
cd sveltekit-frontend
call npm install @urql/svelte@^4.0.5 @urql/core@^5.0.5 @urql/devtools@^2.0.3 @urql/exchange-graphcache@^7.1.2 graphql@^16.8.1

:: GraphQL related devDependencies
echo.
echo Installing GraphQL development dependencies...
call npm install --save-dev @graphql-codegen/cli@^5.0.2 @graphql-codegen/typescript@^4.0.6 @graphql-codegen/typescript-operations@^4.2.0 @graphql-codegen/urql-introspection@^3.0.0

echo.
echo GraphQL dependencies installation completed!
echo.
echo Installed packages:
echo - graphql (Core GraphQL implementation)
echo - graphql-yoga (GraphQL server)
echo - @pothos/core (Schema builder)
echo - @pothos/plugin-drizzle (Drizzle ORM integration)
echo - @urql/svelte (Svelte GraphQL client)
echo - @urql/core (URQL core)
echo - @urql/devtools (Development tools)
echo - @urql/exchange-graphcache (Caching)
echo - @graphql-codegen/* (Code generation tools)
echo.
pause