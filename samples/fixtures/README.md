# Continuity fixtures

Each folder is a story the plot-hole finder is graded against.

```bash
python scripts/grade_fixtures.py                 # all
python scripts/grade_fixtures.py redharbor hound # named
```

| Fixture | Kind | Point |
| --- | --- | --- |
| `ardmore` | synthetic control | Nothing wrong — every finding is a false positive |
| `halberd` | synthetic screenplay | Possession / knowledge / location traps |
| `kestrel` | synthetic prose | Original adversarial lighthouse story |
| `redharbor` | synthetic prose | Destroyed object reused, presence clash, sealed-hold knowledge |
| `frankenstein` | **real** (Gutenberg, trimmed) | Literary control — precision on epistolary prose |
| `hound` | **real** (Gutenberg, trimmed) | Mystery control — open threads are intentional |
| `screw` | **real** (Gutenberg, trimmed) | Ambiguity control — unreliable narration ≠ plot hole |

Real-text fixtures are trimmed to ~900 words/part so a grading run stays affordable. They are precision tests: planted traps are empty; noise is the failure mode.
