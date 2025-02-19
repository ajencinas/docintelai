# 1️⃣ Build Stage: Install dependencies & build Next.js
FROM node:18-alpine AS builder
WORKDIR /app

# ✅ Pass API base URL at build time
ARG NEXT_PUBLIC_API_BASE_URL=https://docintelai-backend.azurewebsites.net
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

# Copy package.json and package-lock.json first (for better caching)
COPY package.json package-lock.json* ./

# Install dependencies (without dev dependencies)
RUN npm ci --omit=dev

# Copy application code
COPY . .

# ✅ Build Next.js in standalone mode
RUN npm run build

# 2️⃣ Production Stage: Minimal final image
FROM node:18-slim AS production
WORKDIR /app

# ✅ Pass API base URL at build time
ARG NEXT_PUBLIC_API_BASE_URL=https://docintelai-backend.azurewebsites.net
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

# ✅ Copy only necessary standalone build files
COPY --from=builder /app/.next/standalone ./ 
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# ✅ Expose the Next.js port
EXPOSE 3000

# ✅ Start the Next.js server
CMD ["node", "server.js"]

