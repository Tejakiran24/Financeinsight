/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
        fontFamily: {
            sans: ['Inter', 'system-ui', 'sans-serif'],
            mono: ['Fira Code', 'monospace'],
        },
        colors: {
            primary: '#00f0ff',    // Neon Cyan
            secondary: '#7b61ff',  // Electric Purple
            accent: '#ff0055',     // Neon Pink
            darkBg: '#09090b',     // Deepest black
            surface: '#18181b',    // Zinc 900
            surfaceGlow: '#27272a' // Zinc 800
        },
        animation: {
            'blob': 'blob 7s infinite',
            'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            'float': 'float 3s ease-in-out infinite',
            'glow': 'glow 2s ease-in-out infinite alternate',
        },
        keyframes: {
            blob: {
                '0%': { transform: 'translate(0px, 0px) scale(1)' },
                '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
                '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
                '100%': { transform: 'translate(0px, 0px) scale(1)' },
            },
            float: {
                '0%, 100%': { transform: 'translateY(0)' },
                '50%': { transform: 'translateY(-10px)' },
            }
        }
    },
  },
  plugins: [],
}
