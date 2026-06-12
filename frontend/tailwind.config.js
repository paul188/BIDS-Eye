/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts}'],
  theme: {
    extend: {
      colors: {
        surface:      '#ffffff',
        panel:        '#ffffff',
        'panel-soft': '#f5f6f7',
        border:       '#e0e0e0',
        accent:       '#1a73e8',
        'accent-hover': '#1765cc',
        ink:          '#202124',
        muted:        '#5f6368',
        'muted-soft': '#80868b',
        'user-bubble':  '#e8f0fe',
        'ai-bubble':    '#ffffff',
      },
    },
  },
  plugins: [],
}
