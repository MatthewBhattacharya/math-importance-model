"""
visualization.py — matplotlib-based plotting and animation.

Two main entry points:
  - plot_snapshot(world, step_num, stats_history)
      Draw the current world state as a static figure with a stats panel.
  - animate(world, params, n_steps, interval_ms, seed)
      Run the simulation while producing a live animation.

Layout (two subplots):
  Left:  Scatter plot of theorems (circles) and mathematicians (triangles).
         Known implications drawn as grey arrows.
         Theorem colour = community importance belief (blue = low, red = high).
         Mathematician colour = ability (cool = low, warm = high).
  Right: Time-series statistics panel:
         - Mean community importance (across all theorems)
         - Number of known implications
         - Number of theorems and mathematicians
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional

from .world import MathematicalWorld
from .params import SimParams
from .dynamics import step as sim_step


# ------------------------------------------------------------------
# Colour helpers
# ------------------------------------------------------------------

def _importance_colour(value: float, vmin: float = -2.0, vmax: float = 2.0) -> str:
    """Map a community importance belief mean to a colour string."""
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdYlBu_r")  # blue=low, red=high importance
    return cmap(norm(np.clip(value, vmin, vmax)))


def _ability_colour(value: float, vmin: float = -2.0, vmax: float = 2.0) -> str:
    """Map an ability value to a colour string."""
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("PiYG")   # pink=low ability, green=high
    return cmap(norm(np.clip(value, vmin, vmax)))


# ------------------------------------------------------------------
# Core drawing function
# ------------------------------------------------------------------

def _draw_world(
    ax: plt.Axes,
    world: MathematicalWorld,
    step_num: int,
    imp_vmin: float = -2.0,
    imp_vmax: float = 2.0,
    ability_vmin: float = -2.0,
    ability_vmax: float = 2.0,
) -> None:
    """Draw the current world state on `ax`."""
    ax.clear()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Mathematical Space  (step {step_num})", fontsize=10)
    ax.set_xlabel("Topic dimension 1")
    ax.set_ylabel("Topic dimension 2")

    imp_norm = matplotlib.colors.Normalize(vmin=imp_vmin, vmax=imp_vmax)
    imp_cmap = plt.get_cmap("RdYlBu_r")
    ab_norm = matplotlib.colors.Normalize(vmin=ability_vmin, vmax=ability_vmax)
    ab_cmap = plt.get_cmap("PiYG")

    # ---- Known implication arrows ----
    for (src_id, tgt_id) in world.known_implications:
        if src_id in world.theorems and tgt_id in world.theorems:
            s = world.theorems[src_id].location
            t = world.theorems[tgt_id].location
            ax.annotate(
                "",
                xy=t, xytext=s,
                arrowprops=dict(
                    arrowstyle="->",
                    color="grey",
                    lw=0.6,
                    alpha=0.5,
                    connectionstyle="arc3,rad=0.05",
                ),
            )

    # ---- Theorems (circles) ----
    for theorem in world.theorems.values():
        imp = world.community_importance(theorem.theorem_id)
        colour = imp_cmap(imp_norm(imp))
        x, y = theorem.location
        ax.scatter(x, y, s=80, c=[colour], marker="o", zorder=3,
                   edgecolors="black", linewidths=0.4)

    # ---- Mathematicians (triangles) ----
    for m in world.mathematicians.values():
        colour = ab_cmap(ab_norm(m.ability))
        x, y = m.location
        ax.scatter(x, y, s=100, c=[colour], marker="^", zorder=4,
                   edgecolors="black", linewidths=0.4)

    # ---- Legend ----
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="○ theorem"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="▲ mathematician"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")


def _draw_stats(
    ax: plt.Axes,
    stats_history: Dict[str, List],
    step_num: int,
) -> None:
    """Draw time-series statistics on `ax`."""
    ax.clear()
    ax.set_title("Statistics", fontsize=10)
    steps = list(range(len(stats_history["mean_importance"])))

    ax.plot(steps, stats_history["mean_importance"], label="mean importance", color="tomato")
    ax.plot(steps, stats_history["n_known_impl"],    label="known implications", color="steelblue")
    ax.plot(steps, stats_history["n_theorems"],      label="theorems", color="seagreen", linestyle="--")
    ax.plot(steps, stats_history["n_mathematicians"], label="mathematicians", color="darkorchid", linestyle=":")
    ax.set_xlabel("step")
    ax.legend(fontsize=7)
    ax.set_xlim(0, max(step_num, 1))


def _collect_stats(world: MathematicalWorld) -> Dict[str, float]:
    """Compute scalar statistics for the current world state."""
    imp_values = [
        world.community_importance(tid) for tid in world.theorems
    ]
    return {
        "mean_importance": float(np.mean(imp_values)) if imp_values else 0.0,
        "n_known_impl": len(world.known_implications),
        "n_theorems": len(world.theorems),
        "n_mathematicians": len(world.mathematicians),
    }


# ------------------------------------------------------------------
# Static snapshot
# ------------------------------------------------------------------

def plot_snapshot(
    world: MathematicalWorld,
    step_num: int = 0,
    stats_history: Optional[Dict[str, List]] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Draw a static figure of the current world state.

    Args:
        world:          The simulation world.
        step_num:       Current step number (for title).
        stats_history:  Dict of lists produced by collect_stats over time.
                        If None, only the world map is shown.
        figsize:        Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    if stats_history is not None:
        fig, (ax_world, ax_stats) = plt.subplots(1, 2, figsize=figsize)
        _draw_world(ax_world, world, step_num)
        _draw_stats(ax_stats, stats_history, step_num)
    else:
        fig, ax_world = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        _draw_world(ax_world, world, step_num)

    # Colourbar for importance
    imp_sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("RdYlBu_r"),
        norm=matplotlib.colors.Normalize(vmin=-2, vmax=2),
    )
    imp_sm.set_array([])
    fig.colorbar(imp_sm, ax=ax_world if stats_history else ax_world,
                 label="community importance belief", fraction=0.03, pad=0.04)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Animation
# ------------------------------------------------------------------

def animate(
    world: MathematicalWorld,
    params: SimParams,
    n_steps: int = 50,
    interval_ms: int = 300,
    seed: int = 0,
) -> FuncAnimation:
    """
    Run the simulation for `n_steps` and produce a live matplotlib animation.

    Args:
        world:       Initial world (will be mutated in-place).
        params:      Simulation parameters.
        n_steps:     Number of steps to animate.
        interval_ms: Milliseconds between frames.
        seed:        RNG seed for reproducibility.

    Returns:
        The FuncAnimation object (keep a reference to prevent garbage collection).

    Usage:
        anim = animate(world, params, n_steps=30)
        plt.show()
    """
    rng = np.random.default_rng(seed)
    stats_history: Dict[str, List] = {
        "mean_importance": [],
        "n_known_impl": [],
        "n_theorems": [],
        "n_mathematicians": [],
    }

    fig, (ax_world, ax_stats) = plt.subplots(1, 2, figsize=(13, 5))
    fig.tight_layout(pad=2.0)

    # Colourbar (created once)
    imp_sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("RdYlBu_r"),
        norm=matplotlib.colors.Normalize(vmin=-2, vmax=2),
    )
    imp_sm.set_array([])
    fig.colorbar(imp_sm, ax=ax_world,
                 label="community importance belief", fraction=0.03, pad=0.04)

    step_counter = [0]  # mutable closure

    def update(_frame: int) -> None:
        sim_step(world, params, rng)
        step_counter[0] += 1

        s = _collect_stats(world)
        for k, v in s.items():
            stats_history[k].append(v)

        _draw_world(ax_world, world, step_counter[0])
        _draw_stats(ax_stats, stats_history, step_counter[0])

    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval_ms, repeat=False)
    return anim


# ------------------------------------------------------------------
# Convenience: run and collect stats without animation
# ------------------------------------------------------------------

def run_and_collect(
    world: MathematicalWorld,
    params: SimParams,
    n_steps: int,
    seed: int = 0,
) -> Dict[str, List]:
    """
    Run the simulation for `n_steps` and return a dict of time-series statistics.

    Useful for headless runs (no display required).
    """
    rng = np.random.default_rng(seed)
    history: Dict[str, List] = {
        "mean_importance": [],
        "n_known_impl": [],
        "n_theorems": [],
        "n_mathematicians": [],
    }
    for _ in range(n_steps):
        sim_step(world, params, rng)
        s = _collect_stats(world)
        for k, v in s.items():
            history[k].append(v)
    return history
