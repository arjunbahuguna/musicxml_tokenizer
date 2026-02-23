# MusicXML Tokenizer

**Course:** Analysis of Symbolic Music and Ethnomusicology  
**Dataset:** [SymbTr v3](https://zenodo.org/records/15470412) — ~3,000 Turkish makam MusicXML files  
**Format:** Jupyter Notebook (`.ipynb`), runs on Google Colab

---

## Assignment

The coding assignment asks for a **symbolic music tokenizer** that converts MusicXML files into sequences of discrete tokens suitable for machine learning models. The table below maps every requirement from the specification to where and how it is satisfied in this project.

| Requirement | Status | Implementation |
|---|---|---|
| Score-to-token (tokenizer) script | ✅ | `tokenize(xml_path) → list[str]` (cell 15) — takes any MusicXML file, returns a flat list of string tokens |
| Token-to-score (de-tokenizer) script | ✅ | `detokenize(tokens) → music21.Score` (cell 20) — reconstructs a playable Score for round-trip evaluation |
| Support any number of parts/instruments | ✅ | Iterates over all `<part>` elements; each emits a `PART_<instrument>` token |
| Support key and time signature changes | ✅ | Mid-score `<key>` and `<time>` changes detected per measure and emitted as new `KEY_` / `TIME_SIG_` tokens |
| Partwise tokenization (every part contains all bars) | ✅ | Outer loop is per-part, inner loop is per-measure — matches MusicXML Partwise layout |
| `.ipynb` notebook that runs on Google Colab | ✅ | Single self-contained notebook; only dependencies are `lxml` and `music21` |

### Required Token Types

All 12 token families from the assignment spec are implemented and validated across 500 files (Section 10 of the notebook):

| Token | Format | Example | Description |
|---|---|---|---|
| `<BOS>` | literal | `<BOS>` | Beginning of sequence |
| `<EOS>` | literal | `<EOS>` | End of sequence |
| `PART_<instrument>` | prefix | `PART_voice` | Part / instrument identifier |
| `CLEF_<type>_<line>` | prefix | `CLEF_G_2` | Clef sign and staff line |
| `PITCH_<note><octave>` | prefix | `PITCH_A_+0.00_4` | Pitch with microtonal cent offset |
| `POS_BAR_<onset>` | prefix | `POS_BAR_0.0` | Beat position within the current bar |
| `POS_ABS_<onset>` | prefix | `POS_ABS_4.0` | Absolute position from score start |
| `DUR_<quarterLength>` | prefix | `DUR_1.0` | Duration in quarter-note units |
| `REST_<type>` | prefix | `REST_quarter` | Rest with MusicXML duration type |
| `BAR_<measure_number>` | prefix | `BAR_1` | Bar / measure boundary marker |
| `TIME_SIG_<num>/<denom>` | prefix | `TIME_SIG_4/4` | Time signature |
| `KEY_<tonic>_<mode>` | prefix | `KEY_C_major` | Key signature |

---

## Version History

The tokenizer was developed iteratively across five versions, each building on the last:

### v0 — Baseline

- Basic `tokenize()` function: BOS/EOS, PART, CLEF, PITCH, POS_BAR, POS_ABS, DUR, REST, BAR, TIME_SIG, KEY
- `detokenize()` for round-trip reconstruction via music21
- Evaluation harness: pitch accuracy, duration accuracy, measure integrity, structural preservation
- 100-file evaluation sample (seed=42)

### v1 — Microtonal Pitch, Ties & Grace Notes

- **Cent-grid pitch encoding**: raw `<alter>` values (Holdrian comma microtones) converted to a cent offset label — 119 unique (step, alter) classes discovered across the corpus
- **Tie tokens**: `TIE_START` / `TIE_STOP` preserve tied-note relationships
- **Grace notes**: `GRACE_<pitch>` tokens emitted without duration contribution

### v2 — Structural Completeness

- **Repeats & endings**: `REPEAT_FWD`, `REPEAT_BWD`, `ENDING_<n>_START`, `ENDING_<n>_STOP` — covers 66 % and 45 % of corpus files respectively
- **Barline styles**: `BARLINE_<style>` for double, final, and other non-regular barlines
- **Fermata**: `FERMATA` token on held notes/rests
- **Tempo**: `TEMPO_<bpm>` extracted from `<sound tempo="...">` — present in 100 % of files
- **Dynamics**: `DYNAMICS_<level>` (pp, p, mp, mf, f, ff, etc.) — present in ~9 % of files
- **Lyrics**: `LYRIC_<text>` for vocal syllables — present in ~85 % of the corpus

### v3 — Efficiency & Vocabulary Optimisation

- **Position quantisation**: beat positions snapped to a grid of 0.25 quarter notes (`POS_GRID = 0.25`), dramatically reducing POS_BAR / POS_ABS vocabulary
- **Rare-pitch merging**: pitches appearing fewer than `RARE_THRESHOLD = 5` times are merged to their nearest cent-grid neighbour, trimming the microtonal pitch vocabulary while preserving perceptual fidelity
- **Lyric closed vocabulary**: syllables below `MIN_LYRIC_FREQ = 3` replaced with `LYRIC_<UNK>`, controlling the long-tail distribution
- **KL divergence** metric added to verify optimised distributions stay close to the originals
- **Distributional analysis**: full-corpus tokenisation pass computes frequency tables before building optimisation mappings

### v4 — Downstream Readiness & Final Polish

- **`REST_<type>` format**: bare `REST` token replaced with `REST_<type>` (e.g. `REST_quarter`, `REST_half`) to match the assignment specification and give downstream models explicit rest-duration information
- **Token-type validation** (Section 10): programmatic check that all 12 required token families plus 9 extended families (`TIE_`, `GRACE_`, `REPEAT_`, `ENDING_`, `BARLINE_`, `FERMATA`, `TEMPO_`, `DYNAMICS_`, `LYRIC_`) are present across a 500-file sample
- **Full-corpus export** (Section 11): all ~2,930 parseable files tokenised; `token2id` / `id2token` mapping built (vocab ≈ 13,000); integer-encoded sequences written to `corpus_tokenized_v4.json` (14 MB) for direct use by a Transformer or other sequence model
- **Cell ordering fix**: notebook verified to execute cleanly top-to-bottom via `nbconvert --execute`

---

## Notebook Structure

| # | Section | Description |
|---|---|---|
| 1 | **Setup & Imports** | `lxml`, `music21`, pandas, matplotlib |
| 2 | **Load Corpus** | Discover ~3,000 XML files, select 100-file eval sample |
| 3 | **Corpus-Wide Pitch Scan** | Build `pitch_map` — 119 unique (step, alter) → cent-label mappings |
| 4 | **v3 Optimisation Defaults** | `POS_GRID`, `pitch_map_v3`, `closed_lyric_vocab` initialised |
| 5a | **Feature Scan** | Corpus statistics: repeats, endings, lyrics, tempo, dynamics prevalence |
| 5b | **XML Structure Exploration** | Inspect a sample file's XML tree |
| 6 | **Helper Functions** | `extract_pitch()`, `extract_duration_ql()`, `extract_barline_tokens()`, `extract_direction_tokens()`, etc. |
| 7 | **`tokenize()`** | Main tokeniser function (v3 with all v4 tweaks) |
| 8 | **Distributional Analysis** | Full-corpus tokenisation → frequency tables |
| 9 | **v3 Optimisation Mappings** | Build `pitch_merge_map`, `closed_lyric_vocab` from distributional data |
| 10 | **`detokenize()`** | Round-trip token → music21.Score reconstruction |
| 11 | **Evaluation Functions** | Pitch/duration accuracy, measure integrity, KL divergence, structural preservation |
| 12 | **Evaluation Run** | 100-file batch eval with v2 → v3 comparison table |
| 13 | **Visualisations** | 2×3 grid: pitch acc, duration acc, vocab reduction, measure integrity, structural preservation, tokens/file |
| 14 | **Error Analysis** | Worst files, error patterns, optimisation impact summary |
| 15 | **Token-Type Validation (v4)** | All 12 required + 9 extended types checked across 500 files |
| 16 | **Full-Corpus Export (v4)** | `corpus_tokenized_v4.json` — integer-encoded sequences + vocab |

---

## Token Format — Example Output

```
[
    "<BOS>",
    "PART_voice",
    "TIME_SIG_4/4",
    "KEY_C_major",
    "CLEF_G_2",
    "TEMPO_120",
    "BAR_1",
    "POS_BAR_0.0", "POS_ABS_0.0", "PITCH_A_+0.00_4", "DUR_1.0",
    "POS_BAR_1.0", "POS_ABS_1.0", "PITCH_B_+0.00_4", "DUR_0.5",
    "POS_BAR_1.5", "POS_ABS_1.5", "REST_quarter",               "DUR_1.0",
    "POS_BAR_2.5", "POS_ABS_2.5", "PITCH_C_+0.00_5", "DUR_1.0", "LYRIC_ya",
    "POS_BAR_3.5", "POS_ABS_3.5", "PITCH_D_+0.00_5", "DUR_0.5", "TIE_START",
    "BAR_2",
    "POS_BAR_0.0", "POS_ABS_4.0", "PITCH_D_+0.00_5", "DUR_0.5", "TIE_STOP",
    ...
    "<EOS>"
]
```

---

## Corpus Statistics (v4)

| Metric | Value |
|---|---|
| Total XML files | ~3,000 |
| Successfully tokenised | 2,930 |
| Parse errors | 70 |
| Total tokens | 3,487,933 |
| Vocabulary size | 13,017 (incl. `<PAD>`) |
| Mean tokens / file | ~1,190 |
| Unique (step, alter) pitch classes | 119 |

---

## Exported Artefact

`corpus_tokenized_v4.json` contains everything needed to train a sequence model:

```json
{
    "vocab":      { "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "..." : "..." },
    "id2token":   { "0": "<PAD>", "1": "<BOS>", "..." : "..." },
    "sequences":  [ [1, 5, 12], ["..."] ],
    "filenames":  [ "acem--ilahi--duyek--...xml", "..." ],
    "metadata":   {
        "n_files": 2930,
        "n_tokens": 3487933,
        "vocab_size": 13017,
        "version": "v4",
        "pos_grid": 0.25,
        "rare_threshold": 5,
        "min_lyric_freq": 3
    }
}
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `lxml` | 6.0.2 | Fast C-based XML parsing (primary tokeniser) |
| `music21` | 9.9.1 | Score reconstruction (detokeniser) & evaluation |
| `pandas` | — | Evaluation DataFrames |
| `matplotlib` | — | Visualisations |

Install (if not on Colab):

```bash
pip install lxml music21 pandas matplotlib
```

---

## Usage

```python
from tokenizer import tokenize  # or run the notebook cell

tokens = tokenize("path/to/file.xml")
print(tokens[:20])
```

Or run the full notebook top-to-bottom in Google Colab or VS Code with a Python 3.12+ kernel.

---

## Repository Branches

| Branch | Description |
|---|---|
| `v0` | Baseline tokenizer + evaluation |
| `v1` | Microtonal pitch, ties, grace notes |
| `v2` | Structural completeness |
| `v3` | Vocabulary optimisation |
| `v4` / `main` | Downstream readiness & final polish |
