#!/bin/bash
# Ubuntu WSL2 Docker Setup Script
# Run this inside Ubuntu WSL2 terminal

echo "🐳 Setting up Docker inside WSL2 Ubuntu..."

# Update package list
echo "📦 Updating package list..."
sudo apt update

# Install Docker and Docker Compose
echo "🔧 Installing Docker..."
sudo apt install docker.io docker-compose -y

# Add current user to docker group
echo "👤 Adding user to docker group..."
sudo usermod -aG docker $USER

# Enable Docker service
echo "🚀 Enabling Docker service..."
sudo systemctl enable docker

echo ""
echo "✅ Docker installation complete!"
echo ""
echo "⚠️  IMPORTANT: You must restart WSL2 for changes to take effect:"
echo "   1. Close this Ubuntu terminal"
echo "   2. In PowerShell run: wsl --shutdown"
echo "   3. Reopen Ubuntu terminal"
echo ""
echo "🧪 Then test Docker with:"
echo "   docker --version"
echo "   docker-compose --version"
echo ""
echo "🚀 To start your project database:"
echo "   cd /mnt/c/path/to/your/project"
echo "   docker-compose up -d"
