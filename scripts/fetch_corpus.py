"""
Fetch long public-domain novels and split them into parts you can ingest.

Short samples do not exercise this system. A five-scene episode never builds a
supersession chain, never puts eight chapters between a fact and the thing that
contradicts it, and never opens a question that stays open for forty thousand
words. The books below were chosen because each one stresses a specific part of
the build, not because they are famous.

Everything here is Project Gutenberg, public domain in the US.

    python scripts/fetch_corpus.py --list
    python scripts/fetch_corpus.py dracula --parts 8
    python scripts/fetch_corpus.py hound --all

Output lands in samples/corpus/<slug>/, which is gitignored. Part 1 is meant to
be pasted into "Start a new story" as the story so far; the rest are pasted
into the composer one at a time.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "gutenberg"
OUT = ROOT / "samples" / "corpus"

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# A chapter stub in a table of contents is a few words long; a real chapter is
# not. This is what separates them without parsing the TOC itself.
MIN_PART_WORDS = 400

# Headings this close together are a contents listing rather than chapters.
TOC_LINE_GAP = 4


@dataclass(frozen=True)
class Book:
    slug: str
    gutenberg_id: int
    title: str
    words: int
    stresses: str
    watch_for: str


CATALOGUE = [
    Book(
        "dracula", 345, "Dracula (Bram Stoker)", 164_000,
        "The fact ledger, harder than anything else here.",
        "Lucy goes alive to ill to dead to walking to at rest: four state "
        "changes on one subject, each superseding the last. Also epistolary, "
        "so every entry is dated and narrated by a different person. If "
        "supersession and visibility hold up on this, they hold up.",
    ),
    Book(
        "moonstone", 155, "The Moonstone (Wilkie Collins)", 198_000,
        "The contradiction adjudicator's false-positive rate.",
        "Narrators disagree with each other on purpose, in good faith, about "
        "events the reader has already seen. Most 'contradictions' here are "
        "not plot holes but unreliable narration, so this measures how often "
        "the checker cries wolf.",
    ),
    Book(
        "hound", 2852, "The Hound of the Baskervilles (Conan Doyle)", 62_000,
        "Dangling questions and payoff debt.",
        "A mystery is a machine for opening questions and paying them off "
        "late. Watch payoff debt climb through the middle chapters and "
        "collapse at the reveal. Stapleton's identity is a supersession "
        "planted twelve chapters before it lands. Best value for its length.",
    ),
    Book(
        "frankenstein", 84, "Frankenstein (Mary Shelley)", 78_000,
        "Scene splitting and POV tracking under a frame story.",
        "Walton narrates Victor narrating the creature. Three nested voices "
        "with no scene headings anywhere, which is the hardest case for "
        "paragraph-boundary segmentation.",
    ),
    Book(
        "jekyll", 43, "Dr Jekyll and Mr Hyde (R L Stevenson)", 29_000,
        "A quick end-to-end run.",
        "Short enough to ingest completely without much cost. The whole plot "
        "is one identity supersession, so it is a clean smoke test of the "
        "ledger before you spend money on Dracula.",
    ),
    Book(
        "woman-in-white", 583, "The Woman in White (Wilkie Collins)", 240_000,
        "Scale, and identity swaps.",
        "The longest here. Two women are deliberately confused with each "
        "other, which is precisely the case where fact subjects collide and "
        "reconciliation has to decide whether two claims are about the same "
        "person.",
    ),
    Book(
        "treasure-island", 120, "Treasure Island (R L Stevenson)", 71_000,
        "The engagement forecast.",
        "Violent swings between long nautical exposition and sudden action, "
        "which is exactly the pacing variance the hazard model claims to "
        "detect. Chapter openings are strong hooks; middles sag.",
    ),
    Book(
        "turn-of-the-screw", 209, "The Turn of the Screw (Henry James)", 43_000,
        "Ambiguity, as an adversarial case.",
        "Nothing in it is reliably true. Fact extraction will assert things "
        "the text refuses to confirm, which is worth seeing: it shows what "
        "the ledger does when a story is built to withhold facts.",
    ),
]

BY_SLUG = {book.slug: book for book in CATALOGUE}

CHAPTER = re.compile(
    r"^\s*(CHAPTER|Chapter|LETTER|Letter)\s+([IVXLCDM]+|\d+)[.\s]*(.*)$"
)

# Some texts number chapters with a bare roman numeral on its own line, with
# the title on the line beneath. Treasure Island does. Matching this everywhere
# would be reckless — a line containing only "I" is not rare — so it is used
# only when the explicit form finds nothing.
BARE_NUMBER = re.compile(r"^\s*([IVXLCDM]{1,7}|\d{1,3})\.?\s*$")

# Last resort, for texts whose sections are titled but never numbered.
ALLCAPS_TITLE = re.compile(r"^\s*([A-Z][A-Z' \-]{6,60})\s*$")


def _fetch(url: str) -> str:
    """
    Fetch over HTTPS, tolerating a machine whose Python trust store is broken.

    `urllib` uses the system trust store and fails here with a self-signed
    certificate in the chain, while `requests` ships certifi and succeeds
    against the same URL. curl is the last resort for the same reason: it has
    its own bundle again.
    """
    try:
        import requests

        response = requests.get(
            url, timeout=60, headers={"User-Agent": "anubhuti-corpus/1.0"}
        )
        response.raise_for_status()
        return response.text
    except ImportError:
        pass
    except Exception as exc:
        print(f"  requests failed ({type(exc).__name__}), falling back to curl")

    finished = subprocess.run(
        ["curl", "-sSL", "--max-time", "120", url],
        capture_output=True, text=True,
    )
    if finished.returncode != 0 or not finished.stdout:
        raise RuntimeError(
            f"could not download {url}: {finished.stderr.strip() or 'empty response'}"
        )
    return finished.stdout


def download(book: Book) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{book.slug}.txt"

    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")

    url = URL.format(id=book.gutenberg_id)
    print(f"  downloading {url}")
    text = _fetch(url)

    cached.write_text(text, encoding="utf-8")
    return text


def strip_boilerplate(text: str) -> str:
    """Drop the Gutenberg licence header and footer."""
    start = re.search(r"\*\*\*\s*START OF TH[EIS]+ PROJECT GUTENBERG[^*]*\*\*\*", text)
    end = re.search(r"\*\*\*\s*END OF TH[EIS]+ PROJECT GUTENBERG[^*]*\*\*\*", text)

    body = text[start.end() if start else 0 : end.start() if end else len(text)]
    return body.strip()


def split_chapters(text: str) -> list[tuple[str, str]]:
    """
    Split on chapter headings, discarding table-of-contents stubs.

    The TOC repeats every heading verbatim, so a naive split doubles the
    chapter count. Rather than trying to find where the TOC ends, anything too
    short to be a chapter is dropped, which handles front matter as well.
    """
    lines = text.splitlines()

    parts = _carve(lines, _drop_toc(lines, _explicit_marks(lines)))
    if len(parts) < 3:
        parts = _carve(lines, _drop_toc(lines, _bare_marks(lines)))
    if len(parts) < 3:
        parts = _carve(lines, _drop_toc(lines, _allcaps_marks(lines)))
    if len(parts) < 2:
        return [("Full text", text)]

    return parts


def _drop_toc(lines: list[str], marks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Remove table-of-contents entries before boundaries are worked out.

    Filtering by body length alone does not work, and the way it fails is
    instructive. A TOC lists every heading with nothing between them, so all
    but the *last* are obviously empty — but the last one runs on into the
    front matter and the first real chapter, so it looks substantial and
    survives, and every real chapter shifts one place. The Moonstone came out
    starting at "Chapter X" for exactly this reason.

    What actually identifies a TOC is geometry: its entries sit on consecutive
    lines. Real chapters are hundreds of lines apart. So a run of three or more
    headings packed within a few lines of each other is a listing, and the
    whole run goes, last entry included.
    """
    if len(marks) < 3:
        return marks

    doomed: set[int] = set()
    run = [0]

    for i in range(1, len(marks)):
        if marks[i][0] - marks[i - 1][0] <= TOC_LINE_GAP:
            run.append(i)
            continue
        if len(run) >= 3:
            doomed.update(run)
        run = [i]

    if len(run) >= 3:
        doomed.update(run)

    return [mark for i, mark in enumerate(marks) if i not in doomed]


def _explicit_marks(lines: list[str]) -> list[tuple[int, str]]:
    marks = []
    for n, line in enumerate(lines):
        match = CHAPTER.match(line)
        if match:
            tail = match.group(3).strip()
            label = f"{match.group(1).title()} {match.group(2)}"
            marks.append((n, f"{label}. {tail}".strip().rstrip(".")))
    return marks


def _bare_marks(lines: list[str]) -> list[tuple[int, str]]:
    """
    A numeral alone on a line, with a title under it or nothing at all.

    Requiring a title was too strict: The Woman in White and The Turn of the
    Screw number their chapters and then go straight into prose, so demanding
    a title found fifteen chapters in a quarter of a million words and glued a
    sentence on as the heading. Untitled numerals are accepted, and the
    minimum-length filter clears up the occasional stray "I".
    """
    marks = []
    for n, line in enumerate(lines):
        match = BARE_NUMBER.match(line)
        if not match:
            continue

        title = ""
        for candidate in lines[n + 1 : n + 4]:
            stripped = candidate.strip()
            if not stripped:
                continue
            # A chapter title is a short phrase without terminal punctuation.
            # A paragraph opening is neither, and gets no title rather than a
            # truncated sentence pretending to be one.
            if len(stripped.split()) <= 9 and not stripped.endswith((".", ",", ";", "!", "?")):
                title = stripped
            break

        marks.append((n, f"{match.group(1)}. {title}".strip().rstrip(".")))
    return marks


def _allcaps_marks(lines: list[str]) -> list[tuple[int, str]]:
    """Sections titled in capitals with no number at all, as in Jekyll and Hyde."""
    return [
        (n, line.strip().title())
        for n, line in enumerate(lines)
        if ALLCAPS_TITLE.match(line)
    ]


def _carve(lines: list[str], marks: list[tuple[int, str]]) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for i, (line_number, heading) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(lines)
        body = "\n".join(lines[line_number + 1 : stop]).strip()
        # Drops table-of-contents stubs and front matter in one move: neither
        # is long enough to be a chapter.
        if len(body.split()) >= MIN_PART_WORDS:
            parts.append((heading, body))
    return parts


def reflow(text: str) -> str:
    """
    Rejoin Gutenberg's hard-wrapped lines into real paragraphs.

    Left as-is, every line break looks like a paragraph boundary to the prose
    segmenter, so a chapter shatters into eighty fragments instead of splitting
    into scenes. The plain-text emphasis markers go too: `_3 May._` reaches
    fact extraction as literal underscores and earns its own noise.
    """
    text = re.sub(r"_([^_\n]{1,120})_", r"\1", text)
    text = text.replace("--", "—")

    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(
        " ".join(line.strip() for line in p.splitlines() if line.strip())
        for p in paragraphs
        if p.strip()
    )


def show_catalogue() -> None:
    print("\nEight public-domain novels, each chosen to break something different.\n")
    for book in CATALOGUE:
        print(f"  \033[1m{book.slug}\033[0m — {book.title}")
        print(f"      {book.words:,} words · stresses: {book.stresses}")
        print(f"      {book.watch_for}\n")
    print("  python scripts/fetch_corpus.py <slug> --parts 8\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?", help="which book")
    parser.add_argument("--list", action="store_true", help="show the catalogue")
    parser.add_argument("--parts", type=int, default=6, help="how many to write")
    parser.add_argument("--all", action="store_true", help="write every chapter")
    args = parser.parse_args()

    if args.list or not args.slug:
        show_catalogue()
        return 0

    book = BY_SLUG.get(args.slug)
    if book is None:
        print(f"Unknown book '{args.slug}'. Try --list.")
        return 1

    print(f"\n{book.title}")
    raw = download(book)
    chapters = split_chapters(strip_boilerplate(raw))
    wanted = chapters if args.all else chapters[: args.parts]

    target = OUT / book.slug
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("part_*.txt"):
        old.unlink()

    print(f"  {len(chapters)} chapters found, writing {len(wanted)}\n")

    total = 0
    for n, (heading, body) in enumerate(wanted, start=1):
        content = f"{heading}\n\n{reflow(body)}\n"
        path = target / f"part_{n:02d}.txt"
        path.write_text(content, encoding="utf-8")
        words = len(content.split())
        total += words
        print(f"  part_{n:02d}.txt  {words:6,} words  {heading[:56]}")

    print(f"\n  {total:,} words in {target.relative_to(ROOT)}")
    print(
        f"\n  Paste part_01 into 'Start a new story' as the story so far, then "
        f"work through the rest in the composer.\n"
        f"  Each part you finalise costs one fact-extraction call plus "
        f"embeddings, so start with three or four.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
