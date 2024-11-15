## To run this project on your computer, you can follow these steps:

### 1. Clone or Download the Project

On GitHub, clone or download this project to your local environment:

```bash
git clone <repository_url>
cd <repository_name>
```
To run the backend, go to the folder `Amazon-Review-AI-Detector`.
To run the frontend, go to the folder `amazon-spam-detector-frontend`.

### 2. Run on Docker

Ensure Docker Desktop is Running. 

Run the following command:

```bash
sudo docker-compose up --build -d
```
OR (for Windows)

```bash
sudo docker-compose up --build -d
```


### 3. Start the Development Server

After running this command, you should see output in the terminal similar to:

```
  VITE v5.x.x  ready in xx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 4. Access the Project in Your Browser

Open your browser and go to the local address shown in the output (usually `http://localhost:5173`). You should be able to see the frontend project running on your local server.

### Explanation of Other Files

- **Dockerfile and docker-compose.yml**: If you want to run this project in a Docker container, you can use these files. Run `docker-compose up` to start the container.
- **tsconfig.json**: TypeScript configuration file, which defines the TypeScript compilation options.
- **eslint.config.js**: ESLint configuration file, used for code style and quality checking.
