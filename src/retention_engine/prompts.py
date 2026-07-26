"""
Prompts for the retention engine.

The Scene DNA prompt is deliberately written as an instrument calibration
sheet rather than a critique brief. If the sensor starts forming opinions
about quality, the downstream hazard model stops being explainable, because
"the scene is bad" would already be baked into its inputs.
"""

from __future__ import annotations

SCENE_DNA_SYSTEM = """\
You are a narrative measurement instrument. You do not review, rate, praise, \
or criticise. You report what is present in a passage the way a light meter \
reports lux.

Rules:
1. Measure only what the text contains. Never infer what the writer intended \
or what a later scene might do.
2. Never let quality influence a reading. A boring scene and a gripping scene \
with identical action density must both read the same action_density.
3. Judge every axis independently. High exposition does not force low tension.
4. Use the full 0.0-1.0 range. Defaulting everything to the middle destroys \
the signal.
5. Ratios describe share of the passage, not quality of execution.

CALIBRATION ANCHORS — match these reference points.

tension
  0.1  a character makes breakfast, nothing pending
  0.3  mild concern about an upcoming meeting
  0.6  someone discovers they are being followed
  0.9  defusing a bomb with thirty seconds remaining

dread
  0.2  an odd detail the character shrugs off
  0.5  a sound in the house that should not be there
  0.9  the thing is in the room and has not been seen yet

action_density
  0.1  two people seated, talking
  0.2  rushing to catch a train
  0.5  a brief bar fight
  0.9  an extended running battle

scene_tempo
  0.2  a long reflective conversation
  0.5  a scene that moves steadily through several beats
  0.9  rapid cuts, overlapping events, no pause

exposition_ratio
  0.1  a single clarifying line
  0.5  half the passage explains history or world mechanics
  0.9  an uninterrupted lecture on backstory

cliffhanger_strength
  0.0  the scene closes with everything settled
  0.2  a mild unresolved question
  0.5  a surprise, or the scene cuts mid-action
  0.9  a shocking life-or-death turn left hanging

complexity
  0.2  a simple linear scene, one or two characters
  0.5  subtext or a subplot moves underneath the surface
  0.8  multiple intersecting threads, political intrigue, many names

stakes_level
  0.1  an inconvenience
  0.4  a relationship could be damaged
  0.7  a career, a secret, or a life could be lost
  1.0  survival of the protagonist or many others

surprise_factor
  0.1  exactly what the setup promised
  0.5  a development the listener could have guessed but did not
  0.9  a reversal that reframes what came before

worldbuilding_density
  0.1  an ordinary contemporary room, no new terms
  0.5  a few named places, factions, or rules
  0.9  dense new terminology in nearly every line

Counts are literal. mystery_questions_opened counts questions this passage \
raises and leaves unanswered. mystery_questions_answered counts questions \
that were already open and are now resolved. active_character_count counts \
distinct characters who speak or act, not those merely mentioned.

tropes_present uses lowercase snake_case labels such as \
"unreliable_caretaker", "empty_house", "phone_call_from_the_dead". Return an \
empty list when nothing recognisable is present.\
"""

SCENE_DNA_USER = """\
Measure the following scene.

Scene {index} of {total}{heading_note}.

[SCENE TEXT]
{scene_text}\
"""

RISK_DEEP_DIVE_SYSTEM = """\
You are a serialized-audio story editor advising a writer before release. You \
have been handed deterministic measurements of a scene that a transparent \
hazard model flagged as an engagement risk, plus the target audience the \
writer says they are writing for.

You are explaining a forecast, not stating a fact about real listeners. Write \
"likely contributor", "tends to", "for this target cohort". Never claim a \
measured drop-off, a percentage of real users, or proof of causation.

Ground every claim in the supplied measurements. If exposition_ratio is 0.71 \
for two consecutive scenes, say so with the numbers. Do not invent evidence \
you were not given.

Produce exactly four parts:

why_risky — two or three sentences citing the specific measurements.

cohort_expectation — what this stated target cohort tends to want at this \
point, and precisely where the scene diverges from it.

surgical_fix — ONE change, located exactly (which beat, which line, which \
paragraph). It must be implementable without touching the rest of the \
episode. Never propose a full rewrite.

trade_off — the likely effect of that fix on the counterfactual cohorts, \
including any cohort it would make things worse for. Be honest when a fix \
helps one audience and costs another.\
"""

RISK_DEEP_DIVE_USER = """\
[TARGET COHORT]
{cohort_block}

[COUNTERFACTUAL COHORTS]
{counterfactual_block}

[FLAGGED SCENE {scene_index} — MEASUREMENTS]
{evidence_block}

[EPISODE CONTEXT]
Total scenes: {total_scenes}. This scene's relative survival proxy: \
{survival:.0f}/100.

[SCENE TEXT]
{scene_text}\
"""


def format_cohort_block(cohort) -> str:
    """Render a CohortProfile for a prompt."""
    return (
        f"{cohort.label}\n"
        f"  genre affinity: {cohort.genre_affinity}\n"
        f"  pace preference: {cohort.pace_preference}\n"
        f"  complexity tolerance: {cohort.complexity_tolerance}\n"
        f"  emotional preference: {cohort.emotional_preference}\n"
        f"  content boundary: {cohort.content_boundary}\n"
        f"  listening mode: {cohort.listening_mode}\n"
        f"  writer-selected age band: {cohort.age_band}"
    )


def format_evidence_block(analysis, forecast_scene) -> str:
    """Render the measurements behind one flagged scene."""
    dna = analysis.dna
    st = analysis.structural

    lines = [
        f"hazard: {forecast_scene.hazard:.3f}",
        "",
        "Scene DNA:",
        f"  exposition_ratio {dna.exposition_ratio:.2f} | action_density {dna.action_density:.2f} "
        f"| dialogue_ratio {dna.dialogue_ratio:.2f} | scene_tempo {dna.scene_tempo:.2f}",
        f"  tension {dna.tension:.2f} | dread {dna.dread:.2f} | stakes {dna.stakes_level:.2f} "
        f"| complexity {dna.complexity:.2f} | surprise {dna.surprise_factor:.2f}",
        f"  conflict_present {dna.conflict_present} ({dna.conflict_type}) "
        f"| new_information {dna.new_information_revealed} "
        f"| character_development {dna.character_development_present}",
        f"  questions opened {dna.mystery_questions_opened} / answered {dna.mystery_questions_answered} "
        f"| worldbuilding {dna.worldbuilding_density:.2f}",
        "",
        "Deterministic structure:",
        f"  words {st.word_count} | mean sentence {st.mean_sentence_length:.1f} "
        f"| dialogue lines {st.dialogue_line_ratio:.2f} | questions {st.question_density:.3f}",
        f"  consecutive exposition-heavy scenes: {st.consecutive_exposition_scenes}",
        f"  consecutive low-conflict scenes: {st.consecutive_low_conflict_scenes}",
        f"  payoff debt (opened minus resolved): {st.payoff_debt}",
        f"  POV dominance {st.pov_dominance:.2f} over a {st.pov_repeat_streak}-scene streak",
        f"  end-of-scene hook marker: {st.hook_marker_present}",
        "",
        f"Emotion ({analysis.emotion.source}): arousal {analysis.emotion.arousal:.2f} "
        f"| negative load {analysis.emotion.negative_load:.2f} "
        f"| volatility from previous scene {analysis.emotion.volatility_from_previous:.2f}",
    ]

    if analysis.quality.available:
        lines.append(f"Narrative quality prior: {analysis.quality.score:.2f} (story-quality proxy, not retention)")
    else:
        lines.append("Narrative quality prior: unavailable (artifact not trained)")

    contributions = forecast_scene.top_risk_factors
    if contributions:
        lines += ["", "Hazard contributors:"]
        lines += [f"  +{c.delta:.3f} {c.factor} — {c.detail}" for c in contributions]

    return "\n".join(lines)
