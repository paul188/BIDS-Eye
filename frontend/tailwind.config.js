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
        // Blue means "link" and nothing else: dataset titles and accession
        // IDs. Buttons, logo and focus rings are neutral. Every text colour
        // passes WCAG AA (4.5:1) on white and on panel-soft.
        surface:        '#ffffff',
        panel:          '#ffffff',
        'panel-soft':   '#f4f5f7',
        border:         '#e3e5e9',
        'border-strong':'#c9cdd4',
        // Outline of inputs and other controls: 3.2:1, the WCAG 1.4.11 minimum is 3:1.
        'border-input': '#8b919d',
        accent:         '#1b1f24',
        'accent-hover': '#000000',
        'accent-soft':  '#eef0f3',
        link:           '#1a40b0',
        'link-visited': '#5b34a6',
        ink:            '#1b1f24',
        muted:          '#555c69',
        'muted-soft':   '#697080',
        'warn-bg':      '#fbf1de',
        'warn-ink':     '#7f5300',
        'danger-bg':    '#fbe9e7',
        'danger-ink':   '#a8261d',
        // "Understood as" chips: filled, so they stand apart from the result
        // text. Teal for structured filters, violet for words matched in the
        // dataset text. Labels pass AA on their fill.
        'chip-bg':        '#d7efe9',
        'chip-ink':       '#1d5b51',
        'chip-text-bg':   '#ece4f6',
        'chip-text-ink':  '#64317f',
      },
    },
  },
  plugins: [],
}
