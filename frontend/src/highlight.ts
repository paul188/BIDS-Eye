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
 */
function phraseTerms(phrase: string, acronymsAlone: boolean): string[] {
  const tokens = (phrase.match(WORD_RE) ?? [])
    .filter(w => w.length > 1 && !STOPWORDS.has(w.toLowerCase()))
  if (tokens.length === 1) {
    const [word] = tokens
    if (COMMON_WORDS.has(word.toLowerCase())) return []
    if (word.length === 2 && !/^[A-Z]{2}$/.test(word)) return []
    return [word]
  }
  const terms = tokens.slice(1).map((word, i) => `${tokens[i]} ${word}`)
  if (acronymsAlone) terms.push(...tokens.filter(isAcronym))
  return terms
}

/** The terms of the chips in effect: their text words, and the labels of coded chips. */
export function highlightTerms(filters: QueryFilter[]): string[] {
  const terms = new Map<string, string>()
  for (const f of filters) {
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

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// Endings a word may add: "Parkinson's", "parkinsonism", "resting", "aged".
// Not any ending: "rest" must not bold "restricted".
const SUFFIX = "(?:['’]s|s|es|ed|d|ing|er|ers|ism|ian)?"
// Between a pair's words: separators and at most one other word ("scanning of one").
const GAP = '[^\\w.!?;:]+(?:\\w+[^\\w.!?;:]+)?'

/**
 * One pattern for all *terms*, case-insensitive except two-letter acronyms
 * ("MS" but not "500 ms"), which land in group 1: check them with isHit.
 */
export function termPattern(terms: string[]): RegExp | null {
  const unique = [...new Set(terms.map(t => t.trim()).filter(Boolean))]
  if (!unique.length) return null
  const short = unique.filter(t => /^[A-Z]{2}$/.test(t))
  const rest = unique.filter(t => !short.includes(t)).sort((a, b) => b.length - a.length)
  const word = (w: string) => escapeRegExp(w) + SUFFIX
  const alternatives = rest.map(t => t.split(' ').map(word).join(GAP))
  // Always a group, so match[1] means the same with or without acronyms.
  const shortGroup = `(${short.length ? short.map(word).join('|') : '(?!)'})`
  return new RegExp(`\\b(?:${[shortGroup, ...alternatives].join('|')})\\b`, 'gi')
}

/** False for a two-letter acronym written in other case ("ms" for "MS"). */
export function isHit(match: RegExpMatchArray, terms: string[]): boolean {
  const short = match[1]
  if (short === undefined) return true
  return terms.some(t => /^[A-Z]{2}$/.test(t) && short.startsWith(t))
}
