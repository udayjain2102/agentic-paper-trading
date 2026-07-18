"""Self-written memory loop: the agent reads its own lessons before deciding,
and writes a dated reflection after each forward tick.

journal/agent_memory.md is the auditable record -- it persists across CI runs
because journal/ lives on the paper-state branch (see paper_cron.sh).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_PATH = "journal/agent_memory.md"
MAX_ENTRIES = 40  # ponytail: hard cap by count, not size; fine while entries stay short


def read_memory(path: str | Path = DEFAULT_PATH) -> str:
    p = Path(path)
    return p.read_text() if p.exists() else ""


def append_reflection(path: str | Path, date_str: str, text: str) -> None:
    """Append a dated entry, then drop the oldest entries past MAX_ENTRIES."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prior = read_memory(p)
    combined = prior + f"\n## {date_str}\n{text}\n"

    parts = combined.split("\n## ")
    preamble, entries = parts[0], parts[1:]  # entries are "<date>\n<text>\n" blocks
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]
    combined = preamble + "\n## " + "\n## ".join(entries)
    p.write_text(combined)


def recent_outcomes(eval_dir: str | Path, bars: dict[str, pd.DataFrame],
                    n_days: int = 10) -> str:
    """Compact text table: recent nonzero/changed positions and next-day realized
    return (pos on day t earns t->t+1 close-to-close, same as backtest.net_returns).
    """
    eval_dir = Path(eval_dir)
    per_sym = {}
    for f in sorted(eval_dir.glob("pos_*.csv")):
        sym = f.stem[len("pos_"):]
        if sym not in bars:
            continue
        pos = pd.read_csv(f, parse_dates=["date"]).set_index("date")["pos"]
        per_sym[sym] = pos
    if not per_sym:
        return ""

    all_dates = sorted(set().union(*(p.index for p in per_sym.values())))
    dates = all_dates[-n_days:]

    lines = []
    for d in dates:
        rows = []
        for sym, pos in per_sym.items():
            if d not in pos.index:
                continue
            cur = float(pos.loc[d])
            idx = pos.index.get_loc(d)
            prev = float(pos.iloc[idx - 1]) if idx > 0 else 0.0
            if cur == 0.0 and cur == prev:
                continue  # flat and unchanged: not a decision worth reporting
            ret = ""
            close = bars[sym]["close"].astype(float)
            if d in close.index:
                loc = close.index.get_loc(d)
                if loc + 1 < len(close):
                    fwd = float(close.iloc[loc + 1] / close.iloc[loc] - 1.0)
                    ret = f" next_day_ret={cur * fwd:+.2%}"
            rows.append(f"{sym} pos={cur:+.0f}{ret}")
        if not rows:
            continue
        if len(rows) > 6:
            n = len(rows)
            rets = [float(r.split("next_day_ret=")[1].rstrip("%")) / 100
                   for r in rows if "next_day_ret=" in r]
            avg = f" avg next-day ret {sum(rets) / len(rets):+.2%}" if rets else ""
            lines.append(f"{d.date()}: held/changed {n} names,{avg}")
        else:
            lines.append(f"{d.date()}: " + "; ".join(rows))
    return "\n".join(lines)


def reflect(complete: Callable[[str], str], memory_path: str | Path,
           outcomes_text: str, date_str: str) -> str:
    """One LLM call: review recent outcomes against prior memory, append a
    dated self-written reflection. Non-fatal on model failure."""
    prior = read_memory(memory_path)
    prompt = (
        "You are reviewing your own recent paper-trading decisions.\n\n"
        f"Prior lessons (your own past reflections):\n{prior or '(none yet)'}\n\n"
        f"Recent decisions and outcomes:\n{outcomes_text or '(no decisions to review)'}\n\n"
        "Write 3-5 short bullet lessons: what worked, what didn't, what to do "
        "differently. Be specific and falsifiable; do not repeat prior lessons "
        "verbatim; if a prior lesson was contradicted by outcomes, say so."
    )
    try:
        text = complete(prompt).strip()
    except Exception:
        return ""
    if not text:
        return ""
    append_reflection(memory_path, date_str, text)
    return text
