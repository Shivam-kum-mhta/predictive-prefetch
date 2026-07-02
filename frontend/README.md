# News Website

This is a simple frontend project for a news website built using React and Vite. The application fetches news articles from an API and displays them in a user-friendly format.

## Project Structure

```
news
├── .gitignore
├── eslint.config.js
├── index.html
├── package.json
├── README.md
├── vite.config.js
├── public
│   └── vite.svg
└── src
    ├── App.css
    ├── App.jsx
    ├── index.css
    ├── main.jsx
    └── assets
        └── react.svg
```

## Features

- Fetches news articles from a public API.
- Displays articles in a list format.
- Responsive design for mobile and desktop views.
- ESLint configured for code quality.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd news
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:3000` to view the application.

## Usage

- The main component is located in `src/App.jsx`.
- Styles for the application can be found in `src/App.css` and `src/index.css`.
- Modify `index.html` to change the main HTML structure if needed.

## License

This project is licensed under the MIT License.