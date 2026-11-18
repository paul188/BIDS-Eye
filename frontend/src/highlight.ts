// Which words of a search to bold in a dataset's description, and the
// pattern that finds them. Mirrors backend/services/query_filters.py
// (highlight_terms), which does the same for README excerpts.
import type { QueryFilter } from './api'

// Postgres's English stopwords: never part of a term.
const STOPWORDS = new Set(`
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be been
being have has had having do does did doing a an the and but if or because as
until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don should now
`.trim().split(/\s+/))

// Too common to bold on their own; fine inside a pair ("behavioural data").
const COMMON_WORDS = new Set([
  'data', 'dataset', 'datasets', 'study', 'task', 'disease', 'disorder', 'patients',
])

const WORD_RE = /[A-Za-z0-9]+/g

function isAcronym(word: string): boolean {
  return /[A-Z]/.test(word.slice(1)) || (/\d/.test(word) && /[A-Za-z]/.test(word))
}

/**
 * Terms of one searched phrase. One word stands alone; several only count
 * as adjacent pairs ("control participants" alone would bold every
 * "participants"). A pair is the two words joined by a space.
 * *acronymsAlone*: acronyms of a coded label ("structural MRI") also stand alone.
 * A digit stays as part of a pair ("7 tesla", "2 back"), never alone.
 */
function phraseTerms(phrase: string, acronymsAlone: boolean): string[] {
  const tokens = (phrase.match(WORD_RE) ?? [])
    .filter(w => (w.length > 1 || /^\d$/.test(w)) && !STOPWORDS.has(w.toLowerCase()))
  if (tokens.length === 1) {
    const [word] = tokens
    if (word.length < 2 || COMMON_WORDS.has(word.toLowerCase())) return []
    // Two letters only as an acronym ("MS", not "ms"); with a digit ("7T", "3D") always.
    if (word.length === 2 && !/^[A-Z]{2}$/.test(word) && !/\d/.test(word)) return []
    return [word]
  }
  const terms = tokens.slice(1).map((word, i) => `${tokens[i]} ${word}`)
  if (acronymsAlone) terms.push(...tokens.filter(isAcronym))
  return terms
}

/**
 * The terms of the chips in effect: their text words, and the labels of coded chips.
 * Every chip kind counts (an age chip carries its group's words: "infant",
 * "newborn"); a negated chip does not, its words are what the dataset must not have.
 */
export function highlightTerms(filters: QueryFilter[]): string[] {
  const terms = new Map<string, string>()
  for (const f of filters) {
    if (f.negated) continue
    const found = f.text_words.flatMap(phrase => phraseTerms(phrase, false))
    // Coded values also show up in descriptions ("fMRI", "EEG", "ADHD").
    if (['modality', 'diagnosis', 'task'].includes(f.kind)) {
      for (const part of f.label.replace(/\s\+\d+$/, '').split(/\s+or\s+/)) {
        found.push(...phraseTerms(part, true))
      }
    }
    for (const term of found) {
      if (!terms.has(term.toLowerCase())) terms.set(term.toLowerCase(), term)
    }
  }
  return [...terms.values()]
}

// ── Field strength ────────────────────────────────────────────────────────
// Mirrors query_filters.highlight_field_strengths / field_strength_pattern:
// a field-strength chip is a number range ("6.5–7.5") with no words, while
// READMEs write "7T", "7 T", "7-Tesla", "3.0T", "1.5 Tesla".

const FIELD_STRENGTH_FIELD = 'Magnetic field strength'
const NOMINAL_TESLA = [0.5, 1, 1.5, 2, 3, 4, 4.7, 7, 9.4, 10.5, 11.7, 14]
const NUMBER = '\\d+(?:\\.\\d+)?'
const RANGE_LABEL_RE = new RegExp(`^(${NUMBER})–(${NUMBER})$`)
const BOUND_LABEL_RE = new RegExp(`^(=|≥|>|≤|<) (${NUMBER})$`)
const STRENGTH_PHRASE_RE = new RegExp(`^\\s*(${NUMBER})\\s*-?\\s*(?:t|tesla)\\s*$`, 'i')

function strengthsOfLabel(label: string): number[] {
  const range = RANGE_LABEL_RE.exec(label)
  if (range) {
    const [low, high] = [Number(range[1]), Number(range[2])]
    return NOMINAL_TESLA.filter(t => low <= t && t <= high)
  }
  const bound = BOUND_LABEL_RE.exec(label)
  if (!bound) return []
  const value = Number(bound[2])
  switch (bound[1]) {
    case '=': return [value]
    case '≥': return NOMINAL_TESLA.filter(t => t >= value)
    case '>': return NOMINAL_TESLA.filter(t => t > value)
    case '≤': return NOMINAL_TESLA.filter(t => t <= value)
    default: return NOMINAL_TESLA.filter(t => t < value)
  }
}

/** Field strengths (tesla) to mark: a field-strength chip's, and searched phrases naming one. */
export function fieldStrengths(filters: QueryFilter[]): number[] {
  const found = new Set<number>()
  for (const f of filters) {
    if (f.negated) continue
    if (f.field === FIELD_STRENGTH_FIELD) strengthsOfLabel(f.label).forEach(t => found.add(t))
    for (const phrase of f.text_words) {
      const named = STRENGTH_PHRASE_RE.exec(phrase)
      if (named) found.add(Number(named[1]))
    }
  }
  return [...found]
}

/** One strength's number: 7 also as "7.0", 1.5 also as "1,5". */
function teslaNumber(value: number): string {
  const [whole, fraction] = String(value).split('.')
  return fraction ? `${whole}[.,]${fraction}0*` : `${whole}(?:\\.0+)?`
}

/** "7T", "7 T", "7-Tesla"; not "17T", "0.7T", "ses-7T", "7 T1w" or "200 T/m/s". */
function strengthAlternative(strengths: number[]): string | null {
  if (!strengths.length) return null
  const numbers = [...new Set(strengths)].sort((a, b) => b - a).map(teslaNumber)
  const space = '[ \\u00a0\\u2009\\u202f]?'
  return `(?<![\\w.,/-])(?:${numbers.join('|')})${space}-?${space}(?:T|[Tt]esla|TESLA)(?![\\w/])`
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// Endings a word may add: "Parkinson's", "parkinsonism", "resting", "aged".
// Not any ending: "rest" must not bold "restricted".
const SUFFIX = "(?:['’]s|s|es|ed|d|ing|er|ers|ism|ian)?"
// Between a pair's words: separators and at most one other word ("scanning of one").
const GAP = '[^\\w.!?;:]+(?:\\w+[^\\w.!?;:]+)?'

/**
 * One pattern for all *terms* and field *strengths*, case-insensitive except
 * two-letter acronyms ("MS" but not "500 ms"), which land in group 1: check
 * them with isHit.
 */
export function termPattern(terms: string[], strengths: number[] = []): RegExp | null {
  const unique = [...new Set(terms.map(t => t.trim()).filter(Boolean))]
  const strength = strengthAlternative(strengths)
  if (!unique.length && !strength) return null
  const short = unique.filter(t => /^[A-Z]{2}$/.test(t))
  const rest = unique.filter(t => !short.includes(t)).sort((a, b) => b.length - a.length)
  const word = (w: string) => escapeRegExp(w) + SUFFIX
  const alternatives = rest.map(t => t.split(' ').map(word).join(GAP))
  // Always a group, so match[1] means the same with or without acronyms.
  const shortGroup = `(${short.length ? short.map(word).join('|') : '(?!)'})`
  const words = `\\b(?:${[shortGroup, ...alternatives].join('|')})\\b`
  // The strength part has no capturing group, so group 1 stays the acronyms'.
  return new RegExp(strength ? `${strength}|${words}` : words, 'gi')
}

/** False for a two-letter acronym written in other case ("ms" for "MS"). */
export function isHit(match: RegExpMatchArray, terms: string[]): boolean {
  const short = match[1]
  if (short === undefined) return true
  return terms.some(t => /^[A-Z]{2}$/.test(t) && short.startsWith(t))
}
