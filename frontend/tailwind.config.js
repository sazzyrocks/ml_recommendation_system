/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        cinema: {
          950: "#090a0f", // Deepest canvas black
          900: "#0f121a", // Background base
          850: "#141823", // Card surface
          800: "#1a1f2e", // Raised container
          750: "#22293d", // Highlight container
          700: "#2c344d", // Border & divider
          600: "#3d486b", // Muted accent border
          500: "#5b6b9e", // Subtext
          400: "#8a97c2", // Muted text
          200: "#c7d2fe", // Light text
          100: "#e0e7ff", // Bright text
        },
        crimson: {
          500: "#e50914", // Classic cinema red
          600: "#b80710", // Deep crimson hover
          400: "#ff2e3b", // Vibrant highlight red
          glow: "rgba(229, 9, 20, 0.4)",
        },
        gold: {
          400: "#fbbf24", // Star rating yellow
          500: "#f59e0b", // Warm rating gold
          600: "#d97706",
        },
        accent: {
          cyan: "#06b6d4",
          violet: "#8b5cf6",
          emerald: "#10b981",
        },
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "-apple-system", "sans-serif"],
        display: ['"Outfit"', '"Inter"', "sans-serif"],
      },
      boxShadow: {
        "card-glow": "0 10px 25px -5px rgba(0, 0, 0, 0.6), 0 0 15px 0 rgba(229, 9, 20, 0.08)",
        "card-hover": "0 20px 35px -10px rgba(0, 0, 0, 0.8), 0 0 25px 2px rgba(229, 9, 20, 0.25)",
        "cyan-glow": "0 0 20px rgba(6, 182, 212, 0.3)",
      },
      backgroundImage: {
        "hero-gradient":
          "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(229, 9, 20, 0.15), transparent 70%), linear-gradient(to bottom, #0f121a, #090a0f)",
        "card-glass":
          "linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%)",
      },
    },
  },
  plugins: [],
};
