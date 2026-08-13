# Week 27 — Exercises

Work top to bottom. Setup (imports, seeds, the lesson's trace corpus, `tiktoken` and
`minbpe` fetch, plotting scaffolds) is given by the notebook; you write only the lines
each exercise asks for. File work this week: E1's tokenizer goes in `bpe.py` with tests
in `tests/test_bpe.py` (acceptance is `pytest`), and E4's fetch script in
`fetch_corpus.py`, writing under `data/`. E5–E6 are the GPU runs: work them on
Colab/Kaggle with `model.py`, `bpe.py`, and `data/corpus.txt` uploaded, then bring the
checkpoint, loss curves, and samples back into this folder — the notebook's checks read
those saved artifacts.

## E1 — BPE from scratch

In `bpe.py`, implement byte-level BPE per lesson §§2–3: `train(text, vocab_size)` with
the 256 bytes as base units and `vocab_size − 256` merges (ties broken by earliest
first occurrence), `encode(text, merges)` applying merges in training order, and
`decode(ids, merges)` concatenating byte sequences with `errors="replace"`.
Hint: lesson §7 is the reference implementation; the changes are the loop bound
(`vocab_size - 256`) and living in a file with its tests.
Accept when: `pytest -q tests/test_bpe.py` is green — `decode(encode(s)) == s` for
every string in the test set, including Unicode physics text (π⁰, √s_NN = 200 GeV).

## E2 — Merge order vs minbpe

Train your `bpe.py` and `minbpe`'s `BasicTokenizer` on the same training text (given
by the notebook) with the same vocab size, recover both ordered merge lists, and
compare them position by position.
Hint: `minbpe` stores merges as a dict `pair -> id`; sort by id to recover training
order.
Accept when: the first 50 merges are identical.

## E3 — Tokenizer autopsy

Tokenize "√s_NN = 200 GeV Au+Au", the numeral list given in the notebook, and a short
code snippet with (a) your corpus-trained BPE and (b) GPT-2's tokenizer via
`tiktoken`; print token counts and the token boundaries for each string.
Hint: `enc = tiktoken.get_encoding("gpt2")`; show boundaries with
`[enc.decode([t]) for t in ids]`.
Accept when: a table of token counts + two identified pitfalls, each stated in one
line.

## E4 — Corpus scrape

In `fetch_corpus.py`, fetch heavy-ion abstracts from the arXiv API (`nucl-ex`,
`nucl-th`, `hep-ex`): page through results, sleep 3 s between calls, retry on empty
pages, deduplicate by arXiv id, and write `data/corpus.txt` plus
`data/provenance.json` recording query, date, and count.
Hint: lesson §5 is the skeleton; your work is the dedup + provenance plumbing, the run
to ≥ 5000, and the re-run comparison.
Accept when: corpus file + provenance exist (≥ 5k abstracts, deduplicated) and a
re-run reproduces the count within API drift.

## E5 — Train nanoGPT on the corpus

On the free GPU: encode the corpus with your tokenizer (vocab 4096), split 90/10 by
position, and train your Week 26 `model.py` at the lesson §6 config (~10 M-class,
≈ 12.3 M by your hand count) with AdamW, LR warmup + cosine decay, gradient clipping
at 1.0, dropout 0.2, and checkpoints every ~500 iters to persistent storage. Compute
the bigram baseline's val loss from train-set pair counts, track train + val loss, and
keep the checkpoint at the val minimum. Demonstrate one resume-from-checkpoint after a
runtime restart (lesson §6).
Hint: bigram val loss = mean of −ln P(next | current) from an add-one-smoothed
(4096, 4096) count table built on train ids; nanoGPT's `train.py` is the pattern for
warmup + cosine and checkpoint/resume.
Accept when: val loss beats the bigram baseline by ≥ 0.5 nats and the loss curve
(train + val) is saved — with the resume demonstrated once.

## E6 — Sample and judge

Sample 10 abstracts (~150 tokens each) at temperature 0.8 from the val-minimum
checkpoint, plus two bigram samples for calibration; save everything to `samples.txt`
and read the transformer's output the way a physicist reads a student's abstract.
Hint: crop the context to the last `ctx` tokens each step; divide the final position's
logits by 0.8, softmax, sample, append, repeat.
Accept when: samples saved and three concrete failure modes named (fabricated
citations, wrong energies, incoherence, …).

## Review

1. (Week 26) Why must loss at init be ≈ ln(vocab_size)? What number does your new
   vocab of 4096 make that, and what did you actually see at step 0?
2. (Week 08) Your run uses AdamW with warmup + cosine decay. Write Adam's update rule
   from memory; what do the two moment estimates track?
3. (Week 04) What does reproducibility require of this GPU run? List the Week 4
   checklist items that apply — seeds, pinned deps, provenance, one-command run — and
   say what each means here.
4. (Weeks 08, 10) The bigram baseline is a giant conditional-probability table. What
   is its maximum-likelihood estimate, and why does add-one smoothing help on val?
