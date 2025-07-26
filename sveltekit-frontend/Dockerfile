# Use a Node.js base image
FROM node:20-alpine

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the application code
COPY . .

# Build the SvelteKit application
RUN npm run build

# Expose the port SvelteKit runs on
EXPOSE 5173

# Set environment variables
ENV PORT=5173
ENV HOST=0.0.0.0

# Start the SvelteKit application
CMD ["node", "build"]