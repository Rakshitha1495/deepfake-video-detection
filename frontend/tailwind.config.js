/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        abyss: "#0A1024",      // page background (deep navy)
        hull: "#101935",       // primary panels
        deck: "#16213F",       // raised surfaces / inputs
        rule: "#24314F",       // borders and dividers
        ink: "#E6ECF8",        // main text
        haze: "#8C99B6",       // secondary text
        signal: "#3FD0E0",     // cyan accent, also "Real"
        pulse: "#9B7BF0",      // violet accent
        alarm: "#F0708A",      // "Fake" / errors
        amber: "#E9B14C",      // warnings
      },
      fontFamily: {
        display: ["Sora", "system-ui", "sans-serif"],
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
      },
      boxShadow: {
        lift: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 12px 32px -18px rgba(3,6,20,0.9)",
      },
    },
  },
  plugins: [],
};
