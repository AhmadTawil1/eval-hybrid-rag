import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from app.chunking import load_chunks
from app.generation import INSUFFICIENT
from eval.benchmark import MODES, RESULTS_PATH, TABLE_COLUMNS, load_collected, load_dataset

CHARTS_DIR = pathlib.Path("eval/charts")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
HEADER_FILL = "#f0efec"
MODE_COLORS = {"dense": "#2a78d6", "sparse": "#eb6834", "hybrid": "#1baf7a"}

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11})


def start(title: str, subtitle: str, grid_axis: str = "y", size=(10, 5.8)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.975, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, subtitle, ha="left", va="top", fontsize=10.5, color=INK_2)
    return fig, ax


def save(fig, name: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / name, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {CHARTS_DIR / name}")


def context_line(results: dict) -> str:
    m = results["models"]
    return f"MultiHop-RAG, k={results['k']} · generator {m['generator']} · judge {m['judge']}"


def grouped_bars(ax, groups, values, labels, ylim, tick_format):
    width, gap = 0.24, 0.02
    x = np.arange(len(groups))
    for i, mode in enumerate(MODES):
        offset = (i - (len(MODES) - 1) / 2) * (width + gap)
        bars = ax.bar(
            x + offset, values[mode], width, color=MODE_COLORS[mode], edgecolor=SURFACE, linewidth=1.5, label=mode
        )
        for rect, text in zip(bars, labels[mode]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=INK_2, fontsize=11)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: tick_format(v)))
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(MODES), frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_2)


def chart_table(results: dict) -> None:
    modes = results["modes"]
    header = ["Mode"] + [name for name, *_ in TABLE_COLUMNS]
    best = {name: (max if higher else min)(value(m) for m in modes.values()) for name, value, _, higher in TABLE_COLUMNS}
    cells, bold = [], set()
    for r, (mode, m) in enumerate(modes.items(), start=1):
        cells.append([mode] + [display(m) for _, _, display, _ in TABLE_COLUMNS])
        for c, (name, value, _, _) in enumerate(TABLE_COLUMNS, start=1):
            if value(m) == best[name]:
                bold.add((r, c))

    fig, ax = plt.subplots(figsize=(12, 2.6), facecolor=SURFACE)
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(list(range(len(header))))
    table.scale(1, 2.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.set_facecolor(HEADER_FILL if r == 0 else SURFACE)
        text = cell.get_text()
        text.set_color(INK_2 if r == 0 else INK)
        if r == 0 or c == 0 or (r, c) in bold:
            text.set_fontweight("bold")
    fig.text(0.01, 0.97, "Retrieval mode comparison", ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.01,
        0.04,
        f"n={results['n']} ({results['n_answerable']} answerable scored by RAGAS, {results['n_trap']} trap), "
        f"k={results['k']}. Bold = best in column.",
        ha="left",
        va="bottom",
        fontsize=10,
        color=INK_2,
    )
    save(fig, "01_comparison_table.png")


def chart_ragas(results: dict) -> None:
    modes = results["modes"]
    metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer relevance", "answer_relevancy"),
        ("Context precision", "context_precision"),
        ("Context recall", "context_recall"),
    ]
    values = {m: [modes[m]["ragas"][key] for _, key in metrics] for m in MODES}
    labels = {m: [f"{v:.3f}" for v in values[m]] for m in MODES}
    fig, ax = start(
        "RAGAS scores by retrieval mode",
        f"Mean over {results['n_answerable']} answerable questions (0 to 1, higher is better) · {context_line(results)}",
    )
    fig.subplots_adjust(top=0.74, bottom=0.1, left=0.07, right=0.98)
    grouped_bars(ax, [name for name, _ in metrics], values, labels, (0, 1), lambda v: f"{v:.1f}")
    save(fig, "02_ragas_metrics.png")


def chart_refusals(results: dict) -> None:
    modes = results["modes"]
    groups = ["Trap questions correctly refused\n(higher is better)", "Answerable questions wrongly refused\n(lower is better)"]
    values, labels = {}, {}
    for m in MODES:
        t, f = modes[m]["trap_refusal"], modes[m]["false_refusal"]
        values[m] = [t["rate"] * 100, f["rate"] * 100]
        labels[m] = [f"{t['refused']}/{t['n']}", f"{f['refused']}/{f['n']}"]
    fig, ax = start(
        "Refusal behaviour by retrieval mode",
        f"Answers of exactly \"{INSUFFICIENT}\" · {context_line(results)}",
        size=(9, 5.8),
    )
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.08, right=0.98)
    grouped_bars(ax, groups, values, labels, (0, 100), lambda v: f"{v:.0f}%")
    save(fig, "03_refusal_behavior.png")


def chart_latency(results: dict) -> None:
    modes = results["modes"]
    values = {m: [modes[m]["latency_ms"]["mean"] / 1000, modes[m]["latency_ms"]["p95"] / 1000] for m in MODES}
    labels = {m: [f"{v:.1f} s" for v in values[m]] for m in MODES}
    top = max(v for vs in values.values() for v in vs) * 1.2
    fig, ax = start(
        "End-to-end latency by retrieval mode",
        f"Retrieval + LLM call per question, {results['n']} questions (lower is better). "
        f"The {results['models']['generator']} call dominates.",
        size=(8, 5.8),
    )
    fig.subplots_adjust(top=0.74, bottom=0.1, left=0.09, right=0.98)
    grouped_bars(ax, ["Mean", "p95"], values, labels, (0, top), lambda v: f"{v:.0f} s")
    save(fig, "04_latency.png")


def evidence_stats(mode: str, gold: dict, text_by_id: dict) -> dict:
    rows = [r for r in load_collected(mode) if r["set"] != "trap" and not r["error"]]
    shares, all_found = [], 0
    outcomes = {"answered": 0, "refused_all": 0, "refused_some": 0, "refused_none": 0}
    for r in rows:
        expected = [text_by_id[c] for c in gold[r["id"]]["expected_chunk_ids"]]
        retrieved = set(r["retrieved_contexts"])
        found = sum(t in retrieved for t in expected)
        shares.append(found / len(expected))
        all_found += found == len(expected)
        if r["response"].strip() != INSUFFICIENT:
            outcomes["answered"] += 1
        elif found == len(expected):
            outcomes["refused_all"] += 1
        elif found == 0:
            outcomes["refused_none"] += 1
        else:
            outcomes["refused_some"] += 1
    return {"mean_share": float(np.mean(shares)), "all_found": all_found, "n": len(rows), "outcomes": outcomes}


def chart_evidence(results: dict, stats: dict) -> None:
    values = {m: [stats[m]["mean_share"] * 100, stats[m]["all_found"] / stats[m]["n"] * 100] for m in MODES}
    labels = {m: [f"{values[m][0]:.0f}%", f"{stats[m]['all_found']}/{stats[m]['n']}"] for m in MODES}
    groups = ["Share of the needed evidence\nfound in the top 5 (average)", "Questions with ALL needed\nevidence found in the top 5"]
    fig, ax = start(
        "How much of the needed evidence retrieval finds",
        "Exact chunk match against the dataset's evidence, so a strict lower bound · "
        f"{stats[MODES[0]]['n']} answerable questions, k={results['k']}",
        size=(9, 5.8),
    )
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.08, right=0.98)
    grouped_bars(ax, groups, values, labels, (0, 100), lambda v: f"{v:.0f}%")
    save(fig, "05_evidence_retrieved.png")


def chart_outcomes(results: dict, stats: dict) -> None:
    segments = [
        ("answered", "Answered", "#c3c2b7", INK),
        ("refused_all", "Refused, all needed evidence was retrieved", "#e87ba4", INK),
        ("refused_some", "Refused, some needed evidence retrieved", "#898781", "#ffffff"),
        ("refused_none", "Refused, none of the needed evidence retrieved", "#52514e", "#ffffff"),
    ]
    fig, ax = start(
        "What happens to the answerable questions",
        f"{stats[MODES[0]]['n']} answerable questions per mode, by outcome and how much evidence retrieval found · "
        f"{context_line(results)}",
        grid_axis="x",
        size=(10.5, 4.8),
    )
    fig.subplots_adjust(top=0.80, bottom=0.30, left=0.09, right=0.98)
    ypos = np.arange(len(MODES))[::-1]
    left = np.zeros(len(MODES))
    for key, label, color, ink in segments:
        counts = np.array([stats[m]["outcomes"][key] for m in MODES])
        ax.barh(ypos, counts, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for y, count, start_x in zip(ypos, counts, left):
            if count:
                ax.text(start_x + count / 2, y, str(count), ha="center", va="center", fontsize=11, fontweight="bold", color=ink)
        left += counts
    ax.set_yticks(ypos)
    ax.set_yticklabels(MODES, color=INK_2, fontsize=12)
    ax.set_xlim(0, stats[MODES[0]]["n"])
    ax.set_xlabel("number of questions", color=MUTED)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_2)
    save(fig, "06_answerable_outcomes.png")


if __name__ == "__main__":
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    chart_table(results)
    chart_ragas(results)
    chart_refusals(results)
    chart_latency(results)

    gold = {i["id"]: i for i in load_dataset()}
    text_by_id = {c["chunk_id"]: c["text"] for c in load_chunks()}
    stats = {m: evidence_stats(m, gold, text_by_id) for m in MODES}
    chart_evidence(results, stats)
    chart_outcomes(results, stats)
