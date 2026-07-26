# The Kestrel Light — what is planted in it

Real novels are internally consistent, which makes them useless for testing a
plot-hole finder: it should find nothing, and if it finds nothing you have
learned nothing. This story is deliberately faulty. Every trap below is
intentional, and the point of the fixture is that you know the answer before
you run it.

Six parts, 2,970 words. Cheap enough to ingest completely.

## How to run it

Paste **Part One** into *Start a new story* as the story so far. Then finalise
parts two through six one at a time in the composer, running **Check the story**
before each finalise.

For the branching traps, fork from Part Three.

## The traps, in the order they become findable

### 1. Supersession chain — four states on one subject

The light itself: **burning nightly** (P1) → **failing, beam falls short** (P2)
→ **out altogether** (P4) → **relit, by someone other than the keeper** (P6).

This is the main test of the fact ledger. After Part Six, `active_facts` should
hold exactly one claim about the light's state, not four. Check the Established
chip: if it still lists "the light burns every night" alongside "the light is
out", supersession is not firing.

### 2. Long-range contradiction — five parts apart

Part One: **Ilse cannot swim**, stated plainly and reinforced with a childhood
near-drowning. Part Six: **Ilse swims out four times and brings in four men.**

Run *Check the story* on Part Six before finalising it. This should be flagged.
It is the clearest true positive in the fixture, and it is deliberately far
enough back that a checker relying on recent context will miss it.

Note the wording in Part Six is hedged — "she could not swim well", "her arms
going anyhow". A good checker still flags it; the hedge is there to stop it
being trivially easy.

### 3. Premature reference

Part Three mentions **"the second key"** in a list of items in the store, as
though the reader already knows about it. It is not established until Part Five,
where it is found sewn into the coat lining.

The premature-reference check is deterministic, so this one should fire
reliably. If it does not, the check is not reaching into the middle of a
sentence.

### 4. Dangling question — opened, never answered

Part Two: **"Who signed the manifest?"** — asked in the text, underlined, and
never resolved anywhere in the six parts.

Should appear under Open threads and should keep payoff debt elevated from Part
Two onward. If payoff debt drops to zero by Part Six, the ledger is closing
questions it has not actually answered.

### 5. Answer without a question

Part Five explains **why the bell on the harbour wall rang for eleven minutes**
on the night Anselm drowned. Nobody ever asked. The text even says so: "She had
not asked why the bell rang."

This is the inverse check, and it is the least likely to fire. Worth watching
because a false negative here is cheap and a false positive is not.

### 6. Near-miss — should NOT be flagged

Part One establishes **Anselm Vary is dead**. Part Four has the harbour master
call the name "Anselm Vary" at a roll and a young man answer "Here."

**This is not a contradiction.** It is his son by a first marriage, made
explicit two lines later. The adjudicator has previously over-fired on exactly
this shape — a permanent fact about a dead person, and a later scene where the
name appears — so this is the false-positive test. If the story check reports a
contradiction here, the prompt has drifted back toward reasoning its way to a
clash.

### 7. Restatement trap

Part Four does both of these things inside one part: it establishes **the ledger
is missing**, and then it **supersedes that** by finding the ledger the same
evening. It also has Ilse write the disappearance into the recovered ledger,
which restates the superseded claim in the same part that ended it.

Expected: after Part Four, the ledger is *present*. If "the ledger is missing"
survives as an active fact, `drop_restatements` is not catching it.

### 8. Branch divergence

**Fork from Part Three.** On the main line the light goes out (P4) and is relit
by Anselm's son (P6). On a fork, write a part where Ilse finds the second key
early and changes the lock.

On that branch, the Part Six draft should now contradict canon — the boy cannot
have a working key — while the same draft passes on main. That difference is the
whole point of branch-scoped canon, and it is the demo worth showing.

## Traps for the other subsystems

### Engagement forecast

**Part Three is the sag.** It is a deliberate history dump: 1798, the brazier,
Colonel Hasketh, the four keepers, the arrangement of the buildings. No
conflict, no dialogue, no character present, no time pressure.

This one has been measured rather than assumed. Running the forecast over all
six parts concatenated, the prose has no scene headings so it splits into
twelve inferred segments, and Part Three's material lands as segments 5 and 6 —
**the two highest-hazard segments in the story at 0.450 each**, against 0.005 to
0.187 for most of the rest. Their evidence:

| | Part Three segments | everywhere else |
| --- | --- | --- |
| exposition ratio | 0.90 | 0.20 – 0.50 |
| scene tempo | 0.20 | 0.30 – 0.60 |
| conflict present | no | mostly yes |

Top hazard factors on the worst segment: `exposition_fatigue` +0.22,
`low_event_movement` +0.141, `exposition_overshoot` +0.135. On the Narrative EKG
this is the deep sag below the cohort's pace line.

If Part Three does not surface as the risk scene, the hazard model is not
detecting the thing it was built to detect.

Because the parts are prose, **they do not map one-to-one onto scenes.** Six
parts became twelve segments. That is expected, and the UI labels them as
inferred.

The Cliffhanger Lab reads the ending as `revelation` with an Unlock Pull Index
around 39. Treat that number as soft: Scene DNA is not reproducible between
runs, and the documented noise band on UPI is ±8.5.

### The screenplay parser

Part Three contains this, buried in the prose:

```
ON THE HARBOUR WALL
The parish maintains a bell, rung in fog, which has hung there since 1871.
```

An all-caps line with a sentence under it. In the Production tab this becomes a
character called ON THE HARBOUR WALL who says that sentence aloud. The editor's
*How this will be read* panel should flag it — `ON` and `THE` are function
words, which is the signal it detects.

**Known limitation this fixture also exposes:** a heading like `JONATHAN
HARKER'S JOURNAL` in the Gutenberg novels contains no function words, so it is
*not* flagged. The detector catches phrases, not every mis-cue.
