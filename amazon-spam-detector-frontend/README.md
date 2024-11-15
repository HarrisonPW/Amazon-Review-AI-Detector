# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type aware lint rules:

- Configure the top-level `parserOptions` property like this:

```js
export default tseslint.config({
  languageOptions: {
    // other options...
    parserOptions: {
      project: ['./tsconfig.node.json', './tsconfig.app.json'],
      tsconfigRootDir: import.meta.dirname,
    },
  },
})
```

- Replace `tseslint.configs.recommended` to `tseslint.configs.recommendedTypeChecked` or `tseslint.configs.strictTypeChecked`
- Optionally add `...tseslint.configs.stylisticTypeChecked`
- Install [eslint-plugin-react](https://github.com/jsx-eslint/eslint-plugin-react) and update the config:

```js
// eslint.config.js
import react from 'eslint-plugin-react'

export default tseslint.config({
  // Set the react version
  settings: { react: { version: '18.3' } },
  plugins: {
    // Add the react plugin
    react,
  },
  rules: {
    // other rules...
    // Enable its recommended rules
    ...react.configs.recommended.rules,
    ...react.configs['jsx-runtime'].rules,
  },
})
```

## To run this frontend project on your computer, you can follow these steps:

### 1. Clone or Download the Project

On GitHub, clone or download this project to your local environment:

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies

The project has a `package.json` file, which means it's a Node.js-based project. Use the following command to install the project dependencies:

```bash
npm install
```

This will read the `package.json` file and install all the required dependencies for the project.

### 3. Start the Development Server

The project uses Vite as the build tool. You can start the Vite development server with this command:

```bash
npm run dev
```

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

If you encounter any errors or need further assistance, feel free to ask!
