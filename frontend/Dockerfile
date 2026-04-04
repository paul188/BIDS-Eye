## Stage 1: build
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

## Stage 2: serve with nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Proxy /api to the backend (see nginx.conf below)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
