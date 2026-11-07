/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        forest: '#145C45',
        action: '#21845F',
        soft: '#E8F3ED',
        paper: '#F7F8F6',
        ink: '#202923',
        muted: '#64716A',
        warn: '#A66A13',
        danger: '#B93838',
        night: '#101713',
        panel: '#19231D',
        elevate: '#202D25',
        primary: '#75C69C',
        light: '#F1F5F2',
        secondary: '#A6B5AA',
        line: '#34443A',
      },
      fontFamily: { sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'] },
    },
  },
  plugins: [],
};
