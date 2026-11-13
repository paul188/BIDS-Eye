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
        // White page, blue links (readers know blue text is clickable),
        // purple once visited. Every text colour passes WCAG AA (4.5:1) on
        // white and on panel-soft; muted-soft fails on accent-soft (4.35).
        surface:        '#ffffff',
        panel:          '#ffffff',
        'panel-soft':   '#f4f5f7',
        border:         '#e3e5e9',
        'border-strong':'#c9cdd4',
        // Outline of inputs and other controls: 3.2:1, the WCAG 1.4.11 minimum is 3:1.
        'border-input': '#8b919d',
        accent:         '#1d4ed8',
        'accent-hover': '#1e3a8a',
        'accent-soft':  '#eaf0fd',
        'link-visited': '#5b34a6',
        ink:            '#1b1f24',
        muted:          '#555c69',
        'muted-soft':   '#697080',
        'warn-bg':      '#fbf1de',
        'warn-ink':     '#7f5300',
        'danger-bg':    '#fbe9e7',
        'danger-ink':   '#a8261d',
      },
    },
  },
  plugins: [],
}
