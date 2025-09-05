#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Database seeding and user creation utility for Legal Case Management System

.DESCRIPTION
    This script provides commands to seed the database with sample data and create individual users.
    It includes comprehensive sample data including users, cases, evidence, and legal documents.

.PARAMETER Command
    The action to perform: 'seed' or 'create-user'

.PARAMETER Email
    Email address for the user (required for create-user)

.PARAMETER Name
    Full name for the user (required for create-user)

.PARAMETER FirstName
    First name for the user (optional)

.PARAMETER LastName
    Last name for the user (optional)

.PARAMETER Role
    Role for the user: prosecutor, detective, analyst, admin (default: prosecutor)

.PARAMETER Password
    Password for the user (default: password123)

.PARAMETER AvatarUrl
    URL for the user's avatar image (optional)

.EXAMPLE
    .\db-seed.ps1 -Command seed
    Seeds the database with comprehensive sample data

.EXAMPLE
    .\db-seed.ps1 -Command create-user -Email "new@example.com" -Name "New User" -Role "prosecutor"
    Creates a single user with prosecutor role

.EXAMPLE
    .\db-seed.ps1 -Command create-user -Email "detective@example.com" -Name "Jane Detective" -Role "detective" -Password "mypassword"
    Creates a detective user with custom password
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("seed", "create-user")]
    [string]$Command,

    [Parameter(Mandatory = $false)]
    [string]$Email = "",

    [Parameter(Mandatory = $false)]
    [string]$Name = "",

    [Parameter(Mandatory = $false)]
    [string]$FirstName = "",

    [Parameter(Mandatory = $false)]
    [string]$LastName = "",

    [Parameter(Mandatory = $false)]
    [ValidateSet("prosecutor", "detective", "analyst", "admin")]
    [string]$Role = "prosecutor",

    [Parameter(Mandatory = $false)]
    [string]$Password = "password123",

    [Parameter(Mandatory = $false)]
    [string]$AvatarUrl = ""
)

# Change to the sveltekit-frontend directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$SvelteKitDir = Join-Path $ScriptDir "sveltekit-frontend"
Set-Location $SvelteKitDir

Write-Host "🔧 Legal Case Management System - Database Utility" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

if ($Command -eq "seed") {
    Write-Host "🌱 Seeding database with comprehensive sample data..." -ForegroundColor Green
    Write-Host ""
    Write-Host "This will create:" -ForegroundColor Yellow
    Write-Host "  • 4 Sample users (prosecutor, detective, analyst, admin)" -ForegroundColor White
    Write-Host "  • 2 Sample criminals" -ForegroundColor White
    Write-Host "  • 2 Sample cases" -ForegroundColor White
    Write-Host "  • 2 Sample evidence items" -ForegroundColor White
    Write-Host "  • 2 Sample legal documents" -ForegroundColor White
    Write-Host "  • 2 Sample notes" -ForegroundColor White
    Write-Host "  • 2 Sample citations" -ForegroundColor White
    Write-Host ""
    
    $confirm = Read-Host "Continue? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "❌ Operation cancelled." -ForegroundColor Red
        exit 1
    }
    
    npm run db:seed
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Database seeded successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔐 Default login credentials:" -ForegroundColor Yellow
        Write-Host "  Prosecutor: prosecutor@example.com / password123" -ForegroundColor White
        Write-Host "  Detective:  detective@example.com / password123" -ForegroundColor White
        Write-Host "  Analyst:    analyst@example.com / password123" -ForegroundColor White
        Write-Host "  Admin:      admin@example.com / password123" -ForegroundColor White
        Write-Host ""
        Write-Host "🚀 You can now start the application with: npm run dev" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Database seeding failed. Check the logs above." -ForegroundColor Red
        exit 1
    }
}
elseif ($Command -eq "create-user") {
    if (-not $Email -or -not $Name) {
        Write-Host "❌ Error: Email and Name are required for user creation." -ForegroundColor Red
        Write-Host ""
        Write-Host "Usage examples:" -ForegroundColor Yellow
        Write-Host "  .\db-seed.ps1 -Command create-user -Email 'user@example.com' -Name 'Full Name'"
        Write-Host "  .\db-seed.ps1 -Command create-user -Email 'det@example.com' -Name 'Detective Smith' -Role 'detective'"
        Write-Host "  .\db-seed.ps1 -Command create-user -Email 'admin@example.com' -Name 'Admin User' -Role 'admin' -Password 'secure123'"
        exit 1
    }
    
    Write-Host "👤 Creating new user..." -ForegroundColor Green
    Write-Host ""
    Write-Host "User details:" -ForegroundColor Yellow
    Write-Host "  Email:     $Email" -ForegroundColor White
    Write-Host "  Name:      $Name" -ForegroundColor White
    Write-Host "  Role:      $Role" -ForegroundColor White
    Write-Host "  Password:  $Password" -ForegroundColor White
    if ($FirstName) { Write-Host "  First Name: $FirstName" -ForegroundColor White }
    if ($LastName) { Write-Host "  Last Name:  $LastName" -ForegroundColor White }
    if ($AvatarUrl) { Write-Host "  Avatar URL: $AvatarUrl" -ForegroundColor White }
    Write-Host ""
    
    $confirm = Read-Host "Create this user? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "❌ Operation cancelled." -ForegroundColor Red
        exit 1
    }
    
    # Create a temporary TypeScript file to create the user
    $TempScript = @"
import { createUser } from './src/lib/server/db/seed.js';

async function createSingleUser() {
    const userData = {
        email: '$Email',
        name: '$Name',
        firstName: '$FirstName',
        lastName: '$LastName',
        role: '$Role',
        password: '$Password',
        avatarUrl: '$AvatarUrl'
    };
    
    const result = await createUser(userData);
    
    if (result.success) {
        console.log('✅ User created successfully!');
        console.log('User ID:', result.user.id);
        console.log('Email:', result.user.email);
        console.log('Name:', result.user.name);
        console.log('Role:', result.user.role);
        console.log('Password:', '$Password');
        process.exit(0);
    } else {
        console.error('❌ User creation failed:', result.message);
        process.exit(1);
    }
}

createSingleUser().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
"@

    $TempFile = "temp-create-user.ts"
    $TempScript | Out-File -FilePath $TempFile -Encoding UTF8
    
    try {
        npx tsx $TempFile
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ User created successfully!" -ForegroundColor Green
            Write-Host "🔐 Login credentials: $Email / $Password" -ForegroundColor Yellow
        } else {
            Write-Host "❌ User creation failed. Check the logs above." -ForegroundColor Red
            exit 1
        }
    } finally {
        if (Test-Path $TempFile) {
            Remove-Item $TempFile
        }
    }
}

Write-Host ""
Write-Host "✅ Database operation completed." -ForegroundColor Green
