FROM node:21-alpine3.17
WORKDIR /frontend

COPY ./frontend/package.json ./

RUN npm install

COPY ./frontend/vite.config.ts .
# COPY tailwind.config.js .
COPY ./frontend/tsconfig.json .
COPY ./frontend/tsconfig.node.json .
# COPY postcss.config.js .

CMD ["npm", "run", "dev"]