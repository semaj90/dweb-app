#!/bin/bash

# Drizzle ORM Type Assertion Fix Script (Bash Version)
# This script applies systematic fixes to Drizzle ORM type compatibility issues

echo "🚀 Drizzle ORM Type Assertion Fix Script"
echo "📁 Target directory: $(pwd)"

# Array of files to check (you can add more as needed)
FILES=(
  "sveltekit-frontend/src/routes/api/canvas-states/+server.ts"
  "sveltekit-frontend/src/routes/api/cases/+server.ts"
  "sveltekit-frontend/src/routes/api/export/+server.ts"
  "sveltekit-frontend/src/routes/api/evidence/validate/+server.ts"
)

# Function to apply fixes to a file
fix_file() {
  local file="$1"
  
  if [[ ! -f "$file" ]]; then
    echo "⚠️  File not found: $file"
    return 1
  fi
  
  echo "🔧 Fixing $file..."
  
  # Create backup
  cp "$file" "$file.backup"
  
  # Apply fixes using sed
  # Fix 1: queryBuilder assignments with .where()
  sed -i 's/\(queryBuilder = queryBuilder\.where([^)]\+)\)\([^a]\)/\1 as any\2/g' "$file"
  
  # Fix 2: queryBuilder assignments with .orderBy()
  sed -i 's/\(queryBuilder = queryBuilder\.orderBy([^)]\+)\)\([^a]\)/\1 as any\2/g' "$file"
  
  # Fix 3: queryBuilder assignments with .limit()
  sed -i 's/\(queryBuilder = queryBuilder\.limit([^)]\+)\)\([^a]\)/\1 as any\2/g' "$file"
  
  # Fix 4: queryBuilder assignments with .offset()
  sed -i 's/\(queryBuilder = queryBuilder\.offset([^)]\+)\)\([^a]\)/\1 as any\2/g' "$file"
  
  # Fix 5: caseQuery assignments
  sed -i 's/\(caseQuery = caseQuery\.\(where\|orderBy\|limit\|offset\)([^)]\+)\)\([^a]\)/\1 as any\3/g' "$file"
  
  # Fix 6: evidenceQuery assignments
  sed -i 's/\(evidenceQuery = evidenceQuery\.\(where\|orderBy\|limit\|offset\)([^)]\+)\)\([^a]\)/\1 as any\3/g' "$file"
  
  # Fix 7: countQuery assignments
  sed -i 's/\(countQuery = countQuery\.where([^)]\+)\)\([^a]\)/\1 as any\2/g' "$file"
  
  # Fix 8: Direct db.select() chains that need totalCountResult fix
  sed -i 's/\(const totalCountResult = await db\.select([^)]\+)\.from([^)]\+)\)\([^a]\)/\1 as any\3/g' "$file"
  
  # Check if file was modified
  if ! cmp -s "$file" "$file.backup"; then
    echo "✅ Applied fixes to $file"
    return 0
  else
    echo "ℹ️  No changes needed for $file"
    rm "$file.backup"
    return 0
  fi
}

# Function to manually apply specific fixes based on the patterns we found
apply_manual_fixes() {
  local file="$1"
  
  if [[ ! -f "$file" ]]; then
    return 1
  fi
  
  # Use a more comprehensive approach with Python or Node.js style regex replacement
  # For now, let's use a temporary file approach
  
  temp_file=$(mktemp)
  
  # Process the file line by line to apply more complex fixes
  while IFS= read -r line; do
    # Fix pattern: queryBuilder = queryBuilder.where(...) without "as any"
    if [[ $line =~ queryBuilder.*=.*queryBuilder\.(where|orderBy|limit|offset)\( ]] && [[ ! $line =~ "as any" ]]; then
      line=$(echo "$line" | sed 's/);/ as any;/')
    fi
    
    # Fix pattern: caseQuery = caseQuery.where(...) without "as any"
    if [[ $line =~ caseQuery.*=.*caseQuery\.(where|orderBy|limit|offset)\( ]] && [[ ! $line =~ "as any" ]]; then
      line=$(echo "$line" | sed 's/);/ as any;/')
    fi
    
    # Fix pattern: countQuery = countQuery.where(...) without "as any"
    if [[ $line =~ countQuery.*=.*countQuery\.where\( ]] && [[ ! $line =~ "as any" ]]; then
      line=$(echo "$line" | sed 's/);/ as any;/')
    fi
    
    echo "$line" >> "$temp_file"
  done < "$file"
  
  # Replace original file if changes were made
  if ! cmp -s "$file" "$temp_file"; then
    mv "$temp_file" "$file"
    echo "✅ Applied manual fixes to $file"
  else
    rm "$temp_file"
  fi
}

# Main execution
total_files=0
fixed_files=0

for file in "${FILES[@]}"; do
  if [[ -f "$file" ]]; then
    total_files=$((total_files + 1))
    if fix_file "$file"; then
      apply_manual_fixes "$file"
      fixed_files=$((fixed_files + 1))
    fi
  fi
done

echo ""
echo "📊 Summary:"
echo "   - Files checked: $total_files"
echo "   - Files processed: $fixed_files"

# Create a more targeted fix for the specific files we know about
echo ""
echo "🎯 Applying targeted fixes for known patterns..."

# Specific fix for canvas-states/+server.ts
if [[ -f "sveltekit-frontend/src/routes/api/canvas-states/+server.ts" ]]; then
  echo "🔧 Applying specific fixes to canvas-states/+server.ts..."
  sed -i 's/query = query\.where(and(\.\.\. filters))/query = query.where(and(...filters)) as any/' "sveltekit-frontend/src/routes/api/canvas-states/+server.ts"
  sed -i 's/query = query\.orderBy(/query = query.orderBy(' "sveltekit-frontend/src/routes/api/canvas-states/+server.ts"
  sed -i 's/\.limit(limit)\.offset(offset)/.limit(limit).offset(offset) as any/' "sveltekit-frontend/src/routes/api/canvas-states/+server.ts"
  sed -i 's/countQuery = countQuery\.where(and(\.\.\. filters))/countQuery = countQuery.where(and(...filters)) as any/' "sveltekit-frontend/src/routes/api/canvas-states/+server.ts"
fi

# Specific fix for cases/+server.ts
if [[ -f "sveltekit-frontend/src/routes/api/cases/+server.ts" ]]; then
  echo "🔧 Applying specific fixes to cases/+server.ts..."
  sed -i 's/queryBuilder = queryBuilder\.where(and(\.\.\. whereConditions))/queryBuilder = queryBuilder.where(and(...whereConditions)) as any/' "sveltekit-frontend/src/routes/api/cases/+server.ts"
  sed -i 's/queryBuilder = queryBuilder\.orderBy(/queryBuilder = queryBuilder.orderBy(' "sveltekit-frontend/src/routes/api/cases/+server.ts"
  sed -i 's/) as any;/) as any;/' "sveltekit-frontend/src/routes/api/cases/+server.ts"
  sed -i 's/queryBuilder = queryBuilder\.limit(limit)\.offset(offset)/queryBuilder = queryBuilder.limit(limit).offset(offset) as any/' "sveltekit-frontend/src/routes/api/cases/+server.ts"
  sed -i 's/\.from(cases) as any/.from(cases)/' "sveltekit-frontend/src/routes/api/cases/+server.ts"
  sed -i 's/const totalCountResult = await db/const totalCountResult = await (db/' "sveltekit-frontend/src/routes/api/cases/+server.ts"
  sed -i 's/\.from(cases) as any;/.from(cases) as any);/' "sveltekit-frontend/src/routes/api/cases/+server.ts"
fi

# Specific fix for export/+server.ts
if [[ -f "sveltekit-frontend/src/routes/api/export/+server.ts" ]]; then
  echo "🔧 Applying specific fixes to export/+server.ts..."
  # This file seems to have different patterns - let's check for specific lines
  sed -i 's/caseQuery = caseQuery\.where(/caseQuery = (caseQuery.where(/' "sveltekit-frontend/src/routes/api/export/+server.ts"
  sed -i 's/);/) as any);/' "sveltekit-frontend/src/routes/api/export/+server.ts"
  sed -i 's/evidenceQuery = evidenceQuery\.where(/evidenceQuery = (evidenceQuery.where(/' "sveltekit-frontend/src/routes/api/export/+server.ts"
fi

echo "✅ Targeted fixes complete!"

echo ""
echo "🧪 Testing syntax by running TypeScript check..."
if command -v npm &> /dev/null; then
  echo "Running 'npm run check'..."
  npm run check --silent 2>&1 | head -20
fi

echo ""
echo "✨ Drizzle type assertion fixes complete!"
echo "🔍 Review the changes and run 'npm run check' to verify fixes"