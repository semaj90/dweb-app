@echo off
echo Running database migration...
cd sveltekit-frontend
call npx drizzle-kit push
cd ..
echo Running database seed...
call npm run db:seed
echo Database setup complete
