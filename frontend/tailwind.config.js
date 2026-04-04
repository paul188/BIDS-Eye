/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts}'],
  theme: {
    extend: {
      colors: {
        surface: '#1e1e2e',
        panel:   '#27273a',
        border:  '#3b3b52',
        accent:  '#7c6af7',
        'accent-hover': '#6b5be6',
        muted:   '#8b8ba7',
        'user-bubble':  '#2d2d45',
        'ai-bubble':    '#232336',
      },
    },
  },
  plugins: [],
}
