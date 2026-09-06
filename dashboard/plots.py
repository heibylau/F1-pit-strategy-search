'''
Builds the two Levin Tree Search result figures (gap plot & stint chart) as
Plotly Figure objects, rendered directly in the Streamlit dashboard via
st.plotly_chart — nothing is written to images/.
'''

import plotly.graph_objects as go

COMPOUND_COLORS = {'MEDIUM': '#e8c73a', 'HARD': '#c9c9c9', 'SOFT': '#e0554f'}
PIT_COLORS      = {'Levin': '#2e7d32', 'Sainz': '#c62828'}


def path_to_stints(path_data: list[dict]) -> list[tuple[int, int, str]]:
    '''
    Derives stints from tire_age discontinuities (a fresh set resets tire_age
    below what normal aging would produce), which also catches same-compound
    pit stops.

    Note: the two race-log generators disagree on which lap carries the
    pit-stop label — race_log.py stamps the *pre*-pit lap with "pit_X" (old
    compound, old tire_age), while F1State.apply_action stamps the *post*-pit
    lap (new compound, tire_age=1). Both agree on tire_age jumping down at the
    first lap of a new stint, so keying off tire_age sidesteps that
    inconsistency instead of trusting the `action` label or `compound` alone.
    '''
    stints = []
    stint_start = path_data[0]['lap']
    current_comp = path_data[0]['compound']
    prev_tire_age = path_data[0]['tire_age']
    for entry in path_data[1:]:
        if entry['tire_age'] != prev_tire_age + 1:
            stints.append((stint_start, entry['lap'] - 1, current_comp))
            stint_start = entry['lap']
            current_comp = entry['compound']
        prev_tire_age = entry['tire_age']
    stints.append((stint_start, path_data[-1]['lap'], current_comp))
    return stints


def build_gapper_plot(path_levin: list[dict], path_sainz: list[dict],
                       traffic_penalties: dict) -> go.Figure:
    laps       = [lap['lap'] for lap in path_levin]
    gaps_levin = [
        l['total_time'] - s['total_time']
        for l, s in zip(path_levin, path_sainz)
    ]

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=PIT_COLORS['Sainz'],
                  line_width=2, annotation_text="Sainz (baseline)",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=laps, y=gaps_levin, mode="lines", name="Levin Tree Search",
        line=dict(color=PIT_COLORS['Levin'], width=2),
        hovertemplate="Lap %{x}<br>Gap: %{y:.2f}s<extra></extra>",
    ))

    for lap_entry in path_levin:
        lap = lap_entry["lap"]
        penalty = traffic_penalties.get(lap, 0.0)
        if penalty > 0.2:
            gap = path_levin[lap - 1]["total_time"] - path_sainz[lap - 1]["total_time"]
            fig.add_annotation(
                x=lap, y=gap, text=f"+{penalty:.1f}s", showarrow=False,
                yshift=18, font=dict(size=10, color=PIT_COLORS['Levin']),
            )

    fig.update_layout(
        title="Lap-by-Lap Time Gap Relative to Carlos Sainz",
        xaxis_title="Lap",
        yaxis_title="Time difference (s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
        hovermode="x unified",
    )
    return fig


def build_stint_chart(path_levin: list[dict], path_sainz: list[dict]) -> go.Figure:
    levin_stints = path_to_stints(path_levin)
    sainz_stints = path_to_stints(path_sainz)

    fig = go.Figure()
    rows = [('Levin', levin_stints), ('Sainz', sainz_stints)]
    seen_compounds = set()

    for label, stints in rows:
        for start, end, comp in stints:
            show_legend = comp not in seen_compounds
            seen_compounds.add(comp)
            fig.add_trace(go.Bar(
                x=[end - start + 1], y=[label], base=[start - 0.5],
                orientation="h",
                marker=dict(color=COMPOUND_COLORS.get(comp, "white"),
                            line=dict(color="gray", width=1)),
                name=comp, legendgroup=comp, showlegend=show_legend,
                hovertemplate=f"{label}: laps {start}-{end}<br>{comp}<extra></extra>",
                width=0.6,
            ))
        for _, pit_lap, _ in stints[:-1]:
            fig.add_vline(
                x=pit_lap + 0.5, line_dash="dash", line_width=2,
                line_color=PIT_COLORS[label],
            )

    for label, color in PIT_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=color, dash="dash", width=2),
            name=f"{label} pit",
        ))

    fig.update_layout(
        title="Pit Stop Strategy",
        xaxis_title="Lap",
        barmode="stack",
        margin=dict(t=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(categoryorder="array", categoryarray=["Sainz", "Levin"]),
    )
    return fig
