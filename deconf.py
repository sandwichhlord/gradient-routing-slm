"""Deconfounding data, per-token category tags, held-out arithmetic and context probes.

The original corpus confounds three things: digits only appear in arithmetic, arithmetic only appears as digits (and
only in the Q/A format), and code never computes. The deconfounding data breaks those links with synthetic generators,
each kept in the domain its surface form belongs to:
  word problems (math)      "three plus four makes seven."           arithmetic without digits
  prose arithmetic (math)   "She added 37 and 48 and got 85."        arithmetic with digits, in prose (not Q/A format)
  number stories (story)    "Mia was 7 and had 3 cats."              digits (and number words) without arithmetic
  code arithmetic (code)    "x = 37 + 48"                            arithmetic inside code
Math deconfounding documents mix word-problem and prose-arithmetic lines.

Tags (one per token, used for evaluation only, never for training):
  0 digit_in_arith_answer         digit tokens after "A:" in a "Q: a op b = A: c" line (Q/A format)
  1 digit_in_story                digit tokens in story-domain text
  2 number_word_in_word_problem   number words (zero..twenty, tens, hundred) in math-domain text
  3 story_word                    other alphabetic tokens in story-domain text
  4 code_keyword_or_indent        Python keywords and whitespace-only (indentation) tokens in code-domain text
  5 digit_in_prose_arith          digit tokens in math-domain text outside Q/A lines (prose arithmetic)
  6 other                         everything else
Format-matched comparisons (prose against prose): digit_in_prose_arith vs digit_in_story, and number words in word
problems vs number words in stories. digit_in_arith_answer differs from stories in format too ("format" comparisons).
Tags depend only on the token, its domain and (for answers) the preceding tokens, so they are the same for real and
synthetic text.

Held-out arithmetic: about 5% of (a, op, b) triples, picked by a fixed hash, never appear in the training math lines
when prepare.py runs with --holdout_arith (the default). eval/failure_trace.py and eval/unlearn.py test on them.
"""
import keyword
import random

import numpy as np

DOMAINS = ["story", "math", "code"]
TAGS = ["digit_in_arith_answer", "digit_in_story", "number_word_in_word_problem", "story_word", "code_keyword_or_indent",
        "digit_in_prose_arith", "other"]
DATA_VERSION = 2   # bump when the generated data or the tags change (part of prepare.py's rebuild signature)
TAG_ID = {t: i for i, t in enumerate(TAGS)}
OTHER = TAG_ID["other"]

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
NUMBER_WORDS = set(UNITS) | {"thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"}

NAMES = ["Lily", "Tom", "Max", "Sara", "Ben", "Mia", "Leo", "Anna", "Sam", "Zoe"]
ANIMALS = ["cat", "dog", "bird", "frog", "bunny", "fox", "bear", "duck"]
THINGS = ["ball", "kite", "box", "hat", "cake", "toy", "book", "flower", "apple", "stone"]
PLACES = ["park", "garden", "forest", "house", "beach", "school", "farm"]
OPS = ["+", "-", "x"]


# ------------------------------------------------------------------------------ arithmetic lines
def is_heldout(a, op, b):
    """A fixed ~5% of problems that training never sees (with --holdout_arith)."""
    return (a * 31 + b * 17 + OPS.index(op) * 7) % 20 == 0


def answer(a, op, b):
    return a + b if op == "+" else a - b if op == "-" else a * b


def random_problem(rng):
    def num():
        digits = rng.randint(1, 3)
        return rng.randint(0, 10 ** digits - 1)
    a, b = num(), num()
    return a, rng.choice(OPS), b


def math_line(rng, holdout=False):
    """One "Q: a op b = A: c" line. With holdout, held-out problems are skipped. Without it the random stream is the
    same as the generator used for the v0-v2 corpus."""
    while True:
        a, op, b = random_problem(rng)
        if not (holdout and is_heldout(a, op, b)):
            return f"Q: {a} {op} {b} = A: {answer(a, op, b)}"


def math_docs_iter(rng, lines_per_doc=20, holdout=False):
    while True:
        yield "\n".join(math_line(rng, holdout) for _ in range(lines_per_doc))


def heldout_problems(n, seed=0):
    """n distinct held-out problems as (prompt, answer string), e.g. ("Q: 37 + 48 = A:", "85")."""
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        a, op, b = random_problem(rng)
        if is_heldout(a, op, b) and (a, op, b) not in seen:
            seen.add((a, op, b))
            out.append((f"Q: {a} {op} {b} = A:", str(answer(a, op, b))))
    return out


# ------------------------------------------------------------------------------ deconfounding generators
def _word_problem_line(rng):
    kind = rng.choice(["plus", "plus", "minus", "times", "apples", "story"])
    if kind == "times":
        a, b = rng.randint(0, 5), rng.randint(0, 4)
        c = a * b
    elif kind == "minus":
        a = rng.randint(0, 20)
        b = rng.randint(0, a)
        c = a - b
    else:
        a = rng.randint(0, 15)
        b = rng.randint(0, 20 - a)
        c = a + b
    A, B, C = UNITS[a], UNITS[b], UNITS[c]
    if kind == "plus":
        return rng.choice([f"{A.capitalize()} plus {B} makes {C}.", f"What is {A} plus {B}? It is {C}."])
    if kind == "minus":
        return rng.choice([f"{A.capitalize()} minus {B} leaves {C}.", f"What is {A} take away {B}? It is {C}."])
    if kind == "times":
        return f"{A.capitalize()} times {B} makes {C}."
    if kind == "apples":
        t = rng.choice(THINGS)
        return f"If you have {A} {t}s and get {B} more, you have {C} {t}s."
    n, t = rng.choice(NAMES), rng.choice(THINGS)
    return f"{n} had {A} {t}s and found {B} more. Now {n} has {C} {t}s."


def word_problem_doc(rng):
    return "\n".join(_word_problem_line(rng) for _ in range(rng.randint(6, 10)))


def prose_arith_line(rng):
    """Arithmetic with digits written as prose, e.g. "Mia added 37 and 48 and got 85." Held-out problems are skipped."""
    while True:
        kind = rng.choice(["+", "+", "-", "x"])
        if kind == "x":
            a, b = rng.randint(0, 12), rng.randint(0, 12)
        elif kind == "-":
            a = rng.randint(0, 99)
            b = rng.randint(0, a)
        else:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
        if not is_heldout(a, kind, b):
            break
    c, n, t = answer(a, kind, b), rng.choice(NAMES), rng.choice(THINGS)
    if kind == "+":
        return rng.choice([f"{n} added {a} and {b} and got {c}.",
                           f"{n} had {a} {t}s and found {b} more, so {n} had {c} {t}s."])
    if kind == "-":
        return rng.choice([f"{n} had {a} {t}s and gave {b} away, which left {c}.",
                           f"{n} took {b} away from {a} and got {c}."])
    return f"When {n} multiplied {a} by {b}, the answer was {c}."


def math_deconf_doc(rng):
    """Word-problem and prose-arithmetic lines, mixed."""
    return "\n".join((_word_problem_line if rng.random() < 0.5 else prose_arith_line)(rng)
                     for _ in range(rng.randint(6, 10)))


def _story_number(rng, style):
    """A number for a story: digits, a number word, or either (style "digits", "words" or "mixed")."""
    if style == "words" or (style == "mixed" and rng.random() < 0.3):
        return UNITS[rng.randint(2, 12)]
    return str(rng.randint(1, 99))


def number_story(rng, style="mixed"):
    n, n2 = rng.sample(NAMES, 2)
    a, t, p = rng.choice(ANIMALS), rng.choice(THINGS), rng.choice(PLACES)
    num = lambda: _story_number(rng, style)  # noqa: E731
    pool = [f"{n} was {num()} years old and had {num()} {a}s.",
            f"Every morning {n} walked {num()} steps to the {p}.",
            f"{n} counted {num()} {t}s in the {p}.",
            f"There were {num()} {t}s on the table, and {n2} took {num()} of them.",
            f"{n} and {n2} played with the {a} for {num()} minutes.",
            f"The {a} was {num()} days old and liked the {p}.",
            f"{n2} gave {n} {num()} red {t}s, and {n} smiled."]
    rng.shuffle(pool)
    return " ".join(pool[:rng.randint(3, 6)]) + " The end."


def code_arith_doc(rng):
    v = rng.sample(["x", "y", "z", "a", "b", "n", "total", "count", "value", "result"], 3)
    num = lambda: rng.randint(0, 99)  # noqa: E731
    op = lambda: rng.choice(["+", "-", "*"])  # noqa: E731
    if rng.random() < 0.5:
        lines = [f"{v[0]} = {num()} {op()} {num()}", f"{v[1]} = {v[0]} {op()} {num()}"]
        if rng.random() < 0.5:
            lines += [f"if {v[1]} > {num()}:", f"    {v[2]} = {v[1]} {op()} {num()}", "else:", f"    {v[2]} = {v[1]}"]
        else:
            lines.append(f"{v[2]} = {v[0]} {op()} {v[1]}")
        lines.append(f"print({v[2]})")
    else:
        fn = rng.choice(["add", "combine", "compute", "step", "update"])
        lines = [f"def {fn}({v[0]}, {v[1]}):", f"    {v[2]} = {v[0]} {op()} {v[1]}", f"    return {v[2]} {op()} {num()}", "",
                 f"print({fn}({num()}, {num()}))"]
    return "\n".join(lines) + "\n"


DECONF_KINDS = {"math": math_deconf_doc, "story": number_story, "code": code_arith_doc}


def deconf_docs_iter(domain, rng):
    gen = DECONF_KINDS[domain]
    while True:
        yield gen(rng)


# ------------------------------------------------------------------------------ tags
_TABLES = {}


def vocab_tables(enc):
    """Per token id: digit-only, number word, alphabetic, Python keyword or indentation, contains a newline."""
    key = enc.name
    if key not in _TABLES:
        n = enc.n_vocab
        digit, numword, alpha, kw_indent, newline = (np.zeros(n, dtype=bool) for _ in range(5))
        keywords = set(keyword.kwlist)
        for i in range(n):
            try:
                s = enc.decode_single_token_bytes(i).decode("utf-8")
            except (UnicodeDecodeError, KeyError):
                continue
            st = s.strip()
            digit[i] = bool(st) and st.isdigit() and st.isascii()
            numword[i] = st.lower() in NUMBER_WORDS
            alpha[i] = bool(st) and st.isalpha() and st.isascii()
            kw_indent[i] = st in keywords or (s != "" and s.strip(" \t") == "" and ("\n" not in s))
            newline[i] = "\n" in s
        a_colon, q_colon = enc.encode_ordinary(" A:"), enc.encode_ordinary("Q:")
        assert len(a_colon) == 2 and len(q_colon) == 2 and a_colon[1] == q_colon[1], (a_colon, q_colon)
        _TABLES[key] = dict(digit=digit, numword=numword, alpha=alpha, kw_indent=kw_indent, newline=newline,
                            a_tok=a_colon[0], q_tok=q_colon[0], colon=a_colon[1], eot=enc.eot_token)
    return _TABLES[key]


def tag_tokens(ids, domain, enc):
    """uint8 tag per token of `ids` (a 1-D array of one domain's text, documents may be concatenated)."""
    t = vocab_tables(enc)
    ids = np.asarray(ids, dtype=np.int64)
    tags = np.full(len(ids), OTHER, dtype=np.uint8)
    if len(ids) == 0:
        return tags
    if domain == "story":
        tags[t["alpha"][ids] & ~t["numword"][ids]] = TAG_ID["story_word"]
        tags[t["digit"][ids]] = TAG_ID["digit_in_story"]
    elif domain == "math":
        pos = np.arange(len(ids))
        start, qline = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
        start[1:] = (ids[1:] == t["colon"]) & (ids[:-1] == t["a_tok"])
        qline[:-1] = (ids[:-1] == t["q_tok"]) & (ids[1:] == t["colon"])       # "Q:" opens a Q/A line
        reset = t["newline"][ids] | (ids == t["eot"])
        last_start = np.maximum.accumulate(np.where(start, pos, -1))
        last_q = np.maximum.accumulate(np.where(qline, pos, -1))
        last_reset = np.maximum.accumulate(np.where(reset, pos, -1))
        in_answer = (last_start > last_reset) & (pos > last_start)
        in_qa = last_q > last_reset
        tags[t["numword"][ids]] = TAG_ID["number_word_in_word_problem"]
        tags[~in_qa & t["digit"][ids]] = TAG_ID["digit_in_prose_arith"]
        tags[in_answer & t["digit"][ids]] = TAG_ID["digit_in_arith_answer"]
    elif domain == "code":
        tags[t["kw_indent"][ids]] = TAG_ID["code_keyword_or_indent"]
    else:
        raise ValueError(domain)
    return tags


# ------------------------------------------------------------------------------ context probes
# (name, context a, context b, kind). "matched": both contexts are prose, so only the function differs (the primary
# comparisons). "format": Q/A lines against prose, where format and function both differ.
PROBE_PAIRS = [("number_words", "word_problem", "story_words", "matched"),
               ("prose_digits", "prose_arith", "story_digits", "matched"),
               ("qa_digits", "arith", "story_digits", "format")]


def build_context_probes(enc, n_seqs=64, seq_len=128, seed=4242):
    """Token windows for eval/context_sensitivity.py: {context: (tokens (N, T) int64, mask (N, T) bool)}. The mask marks
    digit tokens (arith, prose_arith, story_digits) or number-word tokens (word_problem, story_words)."""
    t = vocab_tables(enc)
    rng = random.Random(seed)
    makers = {"arith": (lambda: "\n".join(math_line(rng) for _ in range(20)), "digit"),
              "story_digits": (lambda: number_story(rng, "digits"), "digit"),
              "prose_arith": (lambda: "\n".join(prose_arith_line(rng) for _ in range(8)), "digit"),
              "word_problem": (lambda: word_problem_doc(rng), "numword"),
              "story_words": (lambda: number_story(rng, "words"), "numword")}
    out = {}
    for name, (make, kind) in makers.items():
        rows = []
        for _ in range(n_seqs):
            ids = [enc.eot_token]
            while len(ids) < seq_len:
                ids += enc.encode_ordinary(make()) + [enc.eot_token]
            rows.append(ids[:seq_len])
        tokens = np.array(rows, dtype=np.int64)
        out[name] = (tokens, t[kind][tokens])
    return out
