# Project Anubhuti

An AI writers room for audio drama. Paste in a scene, and a panel of GPT-4o
"experts" critiques it, a synthetic audience predicts minute-by-minute
drop-off, weak minutes get rewritten automatically, and the approved script
is exported as scratch audio with a foley cue manifest.

Story canon lives in Databricks — a Delta table indexed by Vector Search — so
the panel can flag continuity violations against everything you've previously
established.

## How it works

| Stage | Module | What it does |
| --- | --- | --- |
| Lore Engine | `src/lore_engine/` | Embeds canon facts with `text-embedding-3-small`, stores them in a Delta table, retrieves them via Databricks Vector Search |
| Writers Room | `src/writers_room/orchestrator.py` | Sends the scene to `gpt-4o` as a Director, Editor, Psychologist, and Sound Producer, returning a Pydantic-validated `SceneCritique` |
| Audience Simulator | `src/audience_simulator/simulator.py` | Runs three listener personas in parallel and aggregates their per-minute engagement scores into a heatmap |
| Rewrite Engine | `src/writers_room/rewrite_engine.py` | Rewrites only the minutes that scored below threshold, preserving surrounding tone and continuity |
| Audio Engine | `src/audio_engine/synthesizer.py` | Parses the screenplay, casts a `tts-1-hd` voice per character, and emits MP3s plus `production_manifest.json` with foley cues on the timeline |
| Dashboard | `src/dashboard/app.py` | Streamlit UI that wires all of the above together |

The three audience personas are the Impatient Commuter, the Die-Hard Horror
Fan, and the Casual Listener. A minute is flagged as weak when the average
score falls below the threshold **or** when enough individual personas drop
off — so a single enthusiastic persona can't mask a problem by pulling the
average up.

## Setup

Requires Python 3.11+, a Databricks workspace with Unity Catalog and Vector
Search, and an OpenAI API key.

```bash
git clone https://github.com/Aardy-Bond/ZeroToOne.git
cd ZeroToOne

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in your values
```

Authenticate the Databricks CLI (the code reads the `DEFAULT` OAuth profile
from `~/.databrickscfg`, so no PAT is needed):

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

Then provision the Delta table, Vector Search endpoint, and index. This is
idempotent and safe to re-run:

```bash
PYTHONPATH=src python -m lore_engine.setup_vector_db
```

It creates `main.anubhuti.story_lore` and the
`anubhuti-lore-vs-endpoint` Vector Search endpoint. Endpoint provisioning
takes several minutes on first run.

## Running it

```bash
streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. Paste a scene into the editor
and hit **Run Full Analysis**. `samples/death_note_scene.txt` is a ready-made
test scene of roughly three minutes.

Scenes of 400–900 words work best. The simulator buckets the script at about
150 words per minute, so anything shorter produces a heatmap with too few
points to be readable.

Uncheck **Check lore continuity** unless you've ingested canon relevant to the
scene you're testing — otherwise unrelated facts get retrieved at low
relevance and injected into the panel's prompt as if they were established.

### Command line

The full pipeline also runs headless, with flags to skip individual stages:

```bash
PYTHONPATH=src python test_full_pipeline.py --help
```

Individual smoke tests:

```bash
PYTHONPATH=src python test_writers_room.py
PYTHONPATH=src python test_audience_simulator.py
```

## Screenplay formatting

The parser treats any line in all caps as a character cue and reads the lines
beneath it as that character's dialogue. Action lines are analysed but never
spoken.

This means bare all-caps slug lines such as `ON NEWSCAST`, `INSERT HEADLINE:`,
or `BEAT` are misread as speakers, and the action beneath them is dropped.
Rewrite them as ordinary prose and reserve all-caps lines for real character
names.

## Output

`output/` is gitignored and holds generated artifacts:

- `scratch_audio/` — one MP3 per dialogue chunk, named `NNN_character.mp3`
- `production_manifest.json` — every chunk with its start and end timestamp, voice, and any foley trigger landing in that window
- `cue_sheet.txt` — a human-readable timeline of foley cues

## Costs

Every stage calls OpenAI. A single full run on a three-minute scene makes one
embedding call per canon lookup, one `gpt-4o` critique, three parallel
`gpt-4o` persona simulations, an optional rewrite, and one `tts-1-hd` call per
dialogue chunk. Databricks charges separately for the SQL warehouse and the
Vector Search endpoint, which bills while it is running whether or not you are
querying it.
