/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts}'],
  theme: {
    extend: {
      fontFamily: {
        sans:  ['"IBM Plex Sans"', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        serif: ['"IBM Plex Serif"', 'Georgia', 'serif'],
        mono:  ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        // Warm paper background with a deep teal accent. Every text colour
        // here passes WCAG AA (4.5:1) on both `surface` and `panel`.
        surface:        '#fbfaf7',
        panel:          '#ffffff',
        'panel-soft':   '#f2f0ea',
        border:         '#e3dfd6',
        'border-strong':'#cfc9bc',
        // Outline of inputs and other controls: 3.5:1, the WCAG 1.4.11 minimum is 3:1.
        'border-input': '#8a8477',
        accent:         '#0f6b74',
        'accent-hover': '#0b545b',
        'accent-soft':  '#e2efee',
        ink:            '#1d2128',
        muted:          '#575d69',
        'muted-soft':   '#6b7180',
        'warn-bg':      '#fbf1de',
        'warn-ink':     '#7f5300',
        'danger-bg':    '#fbe9e7',
        'danger-ink':   '#a8261d',
      },
    },
  },
  plugins: [],
}
