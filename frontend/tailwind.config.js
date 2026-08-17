/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          blue: "#0072B2",
          orange: "#E69F00",
          green: "#009E73",
        }
      }
    },
  },
  plugins: [],
}
