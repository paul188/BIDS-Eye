// What the page marks in descriptions and generated summaries (src/highlight.ts).
// The README excerpt is marked by the backend (backend/services/results/
// query_filters.py, highlight_terms); both follow the same rules:
// every chip that is not negated, whatever its kind. Run: npm test.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { QueryFilter } from '../src/api/index.ts'
import { fieldStrengths, highlightTerms, isHit, termPattern } from '../src/highlight.ts'

function chip(kind: QueryFilter['kind'], label: string, words: string[], negated = false): QueryFilter {
  return { id: 'w0', kind, field: kind, label, negated, text_words: words, detail: '' }
}

function marked(text: string, filters: QueryFilter[]): string[] {
  const terms = highlightTerms(filters)
  const pattern = termPattern(terms)
  if (!pattern) return []
  return [...text.matchAll(pattern)].filter(m => isHit(m, terms)).map(m => m[0])
}

// The age chip of "infant datasets", as the backend describes it.
const INFANTS = chip('age', 'under 3', [
  'babies', 'baby', 'toddlers', 'toddler', 'neonatal', 'neonates', 'neonate',
  'newborns', 'newborn', 'infant', 'infants',
])

test('an age chip marks the words of its age group', () => {
  assert.deepEqual(
    marked('Every infant was scanned; newborns and toddlers too.', [INFANTS]),
    ['infant', 'newborns', 'toddlers'])
  // Its label is a number range, never a word to mark.
  assert.deepEqual(marked('Children under 3 years.', [INFANTS]), [])
})

test('every chip kind marks its words', () => {
  const species = chip('field', 'mouse', ['mouse', 'mice'])
  const diagnosis = chip('diagnosis', "Parkinson's disease", ['parkinson'])
  assert.deepEqual(
    marked("Mice and Parkinson's patients.", [species, diagnosis]),
    ['Mice', "Parkinson's"])
})

test('a negated chip marks nothing', () => {
  const noSleep = chip('text', '“sleep”', ['sleep'], true)
  assert.deepEqual(highlightTerms([noSleep]), [])
  assert.deepEqual(marked('A sleep study of infants.', [noSleep, INFANTS]), ['infants'])
})

test('older-adult words from hyphenated forms mark as a pair', () => {
  const older = chip('age', '60+', ['older%adults', 'elderly'])
  assert.deepEqual(marked('Older adults and elderly participants; older cars.', [older]),
    ['Older adults', 'elderly'])
})

test('words of the sex and scanner venues are marked; a strength inside a modality chip too', () => {
  const female = chip('sex', 'female', ['female', 'females', 'women'])
  assert.deepEqual(marked('22 women and 20 men; female participants.', [female]), ['women', 'female'])
  const vendor = chip('field', 'recorded', ['ge healthcare', 'general electric', 'signa', 'philips'])
  assert.deepEqual(marked('A GE Healthcare SIGNA and a Philips Achieva.', [vendor]),
    ['GE Healthcare', 'SIGNA', 'Philips'])
  // "7T fMRI" as one condition: labelled by the modality, the strength travels as "7T".
  const fmri7t = chip('modality', 'fMRI', ['7T'])
  assert.deepEqual(fieldStrengths([fmri7t]), [7])
  const pattern = termPattern(highlightTerms([fmri7t]), fieldStrengths([fmri7t]))!
  assert.deepEqual([...'Scanned at 7 T and 7T; not 17T.'.matchAll(pattern)].map(m => m[0]), ['7 T', '7T'])
})
