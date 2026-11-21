"""Parse paragraphs with spaCy; one gz JSON line per sentence: [paper, part, [[text, lemma, pos, dep, head_index, space], ...]]."""
import gzip, json, sys
import spacy
def main(src, dst, field="text"):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.max_length = 3_000_000
    rows = [json.loads(l) for l in open(src)]
    texts = ((r[field][:200000], (r.get("paper", ""), r.get("part", ""))) for r in rows)
    n = 0
    with gzip.open(dst, "wt", compresslevel=4) as fh:
        for doc, (paper, part) in nlp.pipe(texts, as_tuples=True, n_process=7, batch_size=64):
            for s in doc.sents:
                toks = [[t.text, t.lemma_.lower(), t.pos_, t.dep_, t.head.i - s.start, 1 if t.whitespace_ else 0] for t in s]
                if len(toks) < 3: continue
                fh.write(json.dumps([paper, part, toks], separators=(",", ":")) + "\n"); n += 1
    print("sentences", n)
if __name__ == "__main__":
    main(*sys.argv[1:])
