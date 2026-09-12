FROM node:26-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Vite bakes env vars into the build at build time - the browser (not the
# container network) calls this URL, so it must be host-reachable, e.g.
# http://localhost:8000 for local docker-compose, not "http://backend:8000".
# Same convention as project #1's frontend/Dockerfile.
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
