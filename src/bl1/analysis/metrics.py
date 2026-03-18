"""Performance metrics for closed-loop game experiments.

Provides rally-length extraction from game event logs and statistical
comparison across experimental conditions (closed-loop FEP, open-loop,
silent) following the DishBrain analysis methodology.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Rally length extraction
# ---------------------------------------------------------------------------


def rally_length(game_events: list[tuple[float, str]]) -> NDArray:
    """Extract rally lengths from a game event log.

    A rally is the number of consecutive "hit" events before a "miss".
    The returned array has one entry per completed rally (terminated by
    a miss or the end of the recording).

    Args:
        game_events: List of ``(time_ms, event)`` tuples where *event*
            is ``"hit"`` or ``"miss"``.

    Returns:
        1-D int array of rally lengths.  Length 0 if no events.
    """
    if not game_events:
        return np.array([], dtype=np.int32)

    rally_lengths: list[int] = []
    current_rally = 0

    for _time_ms, event in game_events:
        if event == "hit":
            current_rally += 1
        elif event == "miss":
            rally_lengths.append(current_rally)
            current_rally = 0

    # Close any in-progress rally at the end of the event log
    if current_rally > 0:
        rally_lengths.append(current_rally)

    return np.array(rally_lengths, dtype=np.int32)


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------


def performance_comparison(
    results_dict: dict[str, Any],
) -> dict[str, Any]:
    """Compare performance across experimental conditions.

    Runs a basic statistical comparison of rally lengths between
    conditions, using the Mann-Whitney U test (non-parametric, appropriate
    for non-normally-distributed rally lengths).

    Args:
        results_dict: Mapping from condition name to experiment results.
            Each value must be a dict containing a ``"game_events"`` key
            (list of ``(time_ms, event)`` tuples) **or** a
            ``"rally_lengths"`` key (array of rally lengths).

            Example::

                {
                    "closed_loop": results_fep,
                    "open_loop": results_ol,
                    "silent": results_silent,
                }

    Returns:
        Dict with:

        - ``mean_rally``: Dict mapping condition names to mean rally
          length.
        - ``median_rally``: Dict mapping condition names to median rally
          length.
        - ``n_rallies``: Dict mapping condition names to number of
          rallies.
        - ``p_values``: Dict of pairwise ``(cond_a, cond_b) -> p``
          mappings from Mann-Whitney U tests.  Only computed when
          ``scipy`` is available; otherwise an empty dict.
    """
    # --- Extract rally lengths per condition --------------------------------
    rallies: dict[str, NDArray] = {}
    for name, res in results_dict.items():
        if "rally_lengths" in res:
            rl = np.asarray(res["rally_lengths"], dtype=np.float64)
        elif "game_events" in res:
            rl = rally_length(res["game_events"]).astype(np.float64)
        else:
            rl = np.array([], dtype=np.float64)
        rallies[name] = rl

    # --- Summary statistics ------------------------------------------------
    mean_rally: dict[str, float] = {}
    median_rally: dict[str, float] = {}
    n_rallies: dict[str, int] = {}

    for name, rl in rallies.items():
        mean_rally[name] = float(np.mean(rl)) if len(rl) > 0 else 0.0
        median_rally[name] = float(np.median(rl)) if len(rl) > 0 else 0.0
        n_rallies[name] = len(rl)

    # --- Pairwise statistical tests ----------------------------------------
    p_values: dict[tuple[str, str], float] = {}
    condition_names = sorted(rallies.keys())

    try:
        from scipy.stats import mannwhitneyu

        for i, cond_a in enumerate(condition_names):
            for cond_b in condition_names[i + 1 :]:
                ra = rallies[cond_a]
                rb = rallies[cond_b]
                if len(ra) >= 2 and len(rb) >= 2:
                    _stat, p = mannwhitneyu(ra, rb, alternative="two-sided")
                    p_values[(cond_a, cond_b)] = float(p)
                else:
                    p_values[(cond_a, cond_b)] = float("nan")
    except ImportError:
        # scipy not available; skip statistical tests
        pass

    return {
        "mean_rally": mean_rally,
        "median_rally": median_rally,
        "n_rallies": n_rallies,
        "p_values": p_values,
    }
