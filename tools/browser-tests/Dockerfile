FROM node:22-alpine
RUN apk add --no-cache chromium
WORKDIR /test
RUN npm install --no-save playwright@1.56.1
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
