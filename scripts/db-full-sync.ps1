# This script runs the full database synchronization process.

# Generate the database schema from your application code.
Write-Output "Generating database schema..."
npm run db:generate

# Push the schema changes to the database.
Write-Output "Pushing schema to database..."
npm run db:push

# Apply any pending migrations.
Write-Output "Applying database migrations..."
npm run db:migrate

# Start the database studio.
Write-Output "Starting database studio..."
npm run db:studio

Write-Output "Database synchronization complete."
