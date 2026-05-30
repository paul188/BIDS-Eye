/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts}'],
  theme: {
    extend: {
      colors: {
        surface: '#0f1117',
        panel:   '#1c1f2e',
        border:  '#3a3d5c',
        accent:  '#7c6af7',
        'accent-hover': '#6b5be6',
        muted:   '#b0b2cc',
        'user-bubble':  '#1b3354',
        'ai-bubble':    '#1c1f2e',
      },
    },
  },
  plugins: [],
}
