"""
The branch graph: a story's timelines drawn the way a writer thinks about them.

One lane per timeline, one dot per part, left to right in reading order. A
branch's lane begins with a fork line drawn from the last part it inherited, so
you can see at a glance where two versions of the story stopped agreeing.

Inherited parts are drawn hollow on the branch's own lane rather than left
implicit. A branch forked at part 3 really can see parts 1 and 2, and a picture
that omits them invites the reader to think the branch starts from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go

from dashboard import theme

LANE_HEIGHT = 62
MIN_HEIGHT = 190
MAX_HOVER_CHARS = 90


@dataclass
class Lane:
    branch_id: str
    name: str
    row: int
    parent_row: int | None
    forked_at: int | None


def _order_lanes(graph, branches) -> list[Lane]:
    """
    Depth first from each root, so a child sits directly under its parent.

    Breadth first would scatter siblings away from the branch they came from,
    which is the one relationship the picture exists to show.
    """
    by_id = {b.id: b for b in branches}
    rows: dict[str, int] = {}
    lanes: list[Lane] = []

    def walk(branch_id: str) -> None:
        branch = by_id.get(branch_id)
        if branch is None:
            return
        rows[branch_id] = len(lanes)
        lanes.append(
            Lane(
                branch_id=branch_id,
                name=branch.name,
                row=rows[branch_id],
                parent_row=rows.get(branch.parent_id) if branch.parent_id else None,
                forked_at=branch.forked_at,
            )
        )
        for child in graph.children_of(branch_id):
            walk(child)

    for root in graph.roots():
        walk(root)

    # Anything unreachable from a root, which should not happen, still gets drawn.
    for branch in branches:
        if branch.id not in rows:
            walk(branch.id)

    return lanes


def _preview(segment) -> str:
    text = " ".join(segment.text.split())
    if len(text) > MAX_HOVER_CHARS:
        text = text[:MAX_HOVER_CHARS].rsplit(" ", 1)[0] + "..."
    return text


def build(graph, branches, segments, current_branch_id: str | None = None) -> go.Figure:
    """Draw every timeline in the project."""
    lanes = _order_lanes(graph, branches)
    figure = go.Figure()

    own: dict[str, list] = {lane.branch_id: [] for lane in lanes}
    for segment in segments:
        if segment.branch_id in own:
            own[segment.branch_id].append(segment)
    for parts in own.values():
        parts.sort(key=lambda s: s.position)

    for lane in lanes:
        selected = lane.branch_id == current_branch_id
        colour = theme.ACCENT if selected else theme.MUTED
        y = lane.row

        inherited = [
            s
            for s in graph.visible_segments(segments, lane.branch_id)
            if s.branch_id != lane.branch_id
        ]
        mine = own[lane.branch_id]

        # The inherited run, drawn faintly so the eye reads it as borrowed.
        if inherited:
            figure.add_trace(
                go.Scatter(
                    x=[s.position for s in inherited],
                    y=[y] * len(inherited),
                    mode="lines+markers",
                    line={"color": theme.RULE_SOLID, "width": 2},
                    marker={
                        "size": 11,
                        "color": theme.PAPER,
                        "line": {"color": theme.RULE_SOLID, "width": 2},
                    },
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=[
                        f"Part {s.position + 1} · inherited<br>{_preview(s)}"
                        for s in inherited
                    ],
                    showlegend=False,
                )
            )

        # The fork, drawn as a stepped elbow rather than a straight diagonal.
        # Successive forks one lane and one part apart produce diagonals of
        # identical slope, which join up into a single long line and read as
        # one fork spanning the whole graph. A right angle cannot do that.
        if lane.parent_row is not None and lane.forked_at is not None:
            corner = lane.forked_at - 0.5
            figure.add_trace(
                go.Scatter(
                    x=[lane.forked_at - 1, corner, corner, lane.forked_at],
                    y=[lane.parent_row, lane.parent_row, y, y],
                    mode="lines",
                    line={"color": theme.FAINT, "width": 1.5, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if mine:
            # Join the inherited run to this branch's own first part.
            if inherited:
                figure.add_trace(
                    go.Scatter(
                        x=[inherited[-1].position, mine[0].position],
                        y=[y, y],
                        mode="lines",
                        line={"color": colour, "width": 3 if selected else 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            figure.add_trace(
                go.Scatter(
                    x=[s.position for s in mine],
                    y=[y] * len(mine),
                    mode="lines+markers",
                    line={"color": colour, "width": 3 if selected else 2},
                    marker={
                        "size": 15 if selected else 13,
                        "color": colour,
                        "line": {"color": theme.CARD, "width": 2},
                    },
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=[
                        f"<b>{lane.name} · Part {s.position + 1}</b><br>{_preview(s)}"
                        for s in mine
                    ],
                    # Read back on click to switch timelines.
                    meta=lane.branch_id,
                    showlegend=False,
                )
            )
        else:
            # Forked but nothing written yet: an open circle where it will go.
            start = lane.forked_at if lane.forked_at is not None else 0
            figure.add_trace(
                go.Scatter(
                    x=[start],
                    y=[y],
                    mode="markers",
                    marker={
                        "size": 13,
                        "color": theme.PAPER,
                        "line": {"color": colour, "width": 2},
                        "symbol": "circle",
                    },
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=[f"<b>{lane.name}</b><br>Nothing written here yet"],
                    meta=lane.branch_id,
                    showlegend=False,
                )
            )

    positions = [s.position for s in segments] or [0]
    last = max(positions)

    figure.update_layout(
        height=max(MIN_HEIGHT, MIN_HEIGHT // 2 + LANE_HEIGHT * len(lanes)),
        # The left margin is set by the longest timeline name, which the writer
        # chooses, so it has to be measured rather than guessed.
        margin={"r": 24, "t": 18, "b": 34},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={
            "bgcolor": theme.CARD,
            "bordercolor": theme.RULE_SOLID,
            "font": {"color": theme.INK, "size": 12},
            "align": "left",
        },
        clickmode="event+select",
        dragmode=False,
        xaxis={
            "title": "",
            "tickmode": "array",
            "tickvals": list(range(last + 1)),
            "ticktext": [f"Part {p + 1}" for p in range(last + 1)],
            "showgrid": True,
            "gridcolor": theme.RULE_SOLID,
            "gridwidth": 1,
            "zeroline": False,
            "range": [-0.6, last + 0.6],
            "tickfont": {"color": theme.FAINT, "size": 11},
            "fixedrange": True,
        },
        yaxis={
            "tickmode": "array",
            "tickvals": [lane.row for lane in lanes],
            "ticktext": [
                f"<b>{lane.name}</b>"
                if lane.branch_id == current_branch_id
                else lane.name
                for lane in lanes
            ],
            "showgrid": False,
            "zeroline": False,
            "autorange": "reversed",
            "tickfont": {"color": theme.MUTED, "size": 12},
            "fixedrange": True,
            "automargin": True,
            "ticklabelposition": "outside",
        },
    )

    return figure


def clicked_branch(event, graph_figure: go.Figure) -> str | None:
    """
    Which timeline the writer clicked, if any.

    Plotly hands back a curve index rather than anything meaningful, so the
    branch id rides along in each trace's `meta` and is looked up here.
    """
    if event is None:
        return None

    points = getattr(getattr(event, "selection", None), "points", None) or []
    for point in points:
        index = point.get("curve_number")
        if index is None:
            continue
        try:
            meta = graph_figure.data[index].meta
        except (IndexError, AttributeError):
            continue
        if isinstance(meta, str) and meta:
            return meta

    return None


def legend() -> str:
    return (
        f"<div style='font-size:.78rem;color:{theme.FAINT};margin-top:-.4rem'>"
        f"<span style='color:{theme.ACCENT}'>●</span> the timeline you are on "
        f"&nbsp;&nbsp;<span style='color:{theme.MUTED}'>●</span> written on that "
        f"timeline &nbsp;&nbsp;<span style='color:{theme.RULE_SOLID}'>○</span> inherited "
        f"from before the fork</div>"
    )
