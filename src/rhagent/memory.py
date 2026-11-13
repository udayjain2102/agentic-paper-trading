"""Self-written memory loop: the agent reads its own lessons before deciding,
and rewrites them after each forward tick.

journal/agent_memory.md holds ONE current lessons block -- the only memory a
prompt ever sees. Every block ever written is also appended to
journal/agent_memory_archive.md, which is the audit trail and is never read
back into a prompt. Both persist across CI runs because journal/ lives on the
paper-state branch (see paper_cron.sh).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_PATH = "journal/agent_memory.md"
MAX_BULLETS = 5
MAX_BULLET_CHARS = 220


def archive_path(path: str | Path = DEFAULT_PATH) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}_archive{p.suffix}")


def read_memory(path: str | Path = DEFAULT_PATH) -> str:
    """The current lessons block -- the newest dated entry, nothing else. A
    file still in the old append-forever format yields only its last entry."""
    p = Path(path)
    if not p.exists():
        return ""
    entries = p.read_text().split("\n## ")[1:]
    return "\n## " + entries[-1] if entries else ""


def write_lessons(path: str | Path, date_str: str, text: str) -> None:
    """Replace the current block with a dated one and append it to the archive.

    First write against a pre-archive file moves that whole file (the old
    dated log) into the archive first, so nothing already written is lost.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arch = archive_path(p)
    legacy = p.read_text() if p.exists() and not arch.exists() else ""
    entry = f"\n## {date_str}\n{text}\n"
    with arch.open("a") as f:
        f.write(legacy + entry)
    p.write_text(entry)


def _bullets(text: str) -> str:
    """Keep only bullet lines, one line each, capped in count and length --
    the model's preamble, headings and sub-paragraphs never reach the file."""
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*[-*\u2022]\s+(.*\S)", line)
        if m:
            out.append("- " + " ".join(m.group(1).split())[:MAX_BULLET_CHARS])
    return "\n".join(out[:MAX_BULLETS])


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
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        if "status" in df:
            # A failed call holds the prior position: that is an outage, not a
            # decision, and a lesson drawn from it is a lesson about nothing.
            df = df[df["status"] == "ok"]
        per_sym[sym] = df["pos"]
    if not per_sym:
        return ""

    all_dates = sorted(set().union(*(p.index for p in per_sym.values())))
    dates = all_dates[-n_days:]

    lines = []
    setups: dict[str, list[float]] = {}  # the agent's own inputs -> realized P&L
    for d in dates:
        rows, obs = [], []  # obs: (pos, pnl) for realized nonzero positions
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
                    pnl = cur * float(close.iloc[loc + 1] / close.iloc[loc] - 1.0)
                    ret = f" next_day_ret={pnl:+.2%}"
                    if cur != 0.0:
                        obs.append((cur, pnl))
                        # same 5-bar momentum the decision prompt showed (engine._mom5)
                        mom = close.iloc[loc] / close.iloc[max(loc - 5, 0)] - 1.0
                        setups.setdefault(
                            f"{'long' if cur > 0 else 'short'} with momentum_5d "
                            f"{'up' if mom > 0 else 'down'}", []).append(pnl)
            rows.append(f"{sym} pos={cur:+.0f}{ret}")
        if not rows:
            continue
        if len(rows) > 6:
            # Split by the side the agent chose: a single basket average hides
            # the only thing it controls.
            sides = (("long", [p for c, p in obs if c > 0]),
                     ("short", [p for c, p in obs if c < 0]))
            parts = [f"{name} {len(v)} names avg next_day_ret={sum(v) / len(v):+.2%}"
                     for name, v in sides if v]
            lines.append(f"{d.date()}: " + ("; ".join(parts) or f"{len(rows)} names, not yet realized"))
        else:
            lines.append(f"{d.date()}: " + "; ".join(rows))
    if setups:
        lines.append("By setup over these days (per position, next-day):")
        lines += [f"  {k}: n={len(v)} avg {sum(v) / len(v):+.2%} "
                  f"win_rate {sum(x > 0 for x in v) / len(v):.0%}"
                  for k, v in sorted(setups.items())]
    return "\n".join(lines)


def reflect(complete: Callable[[str], str], memory_path: str | Path,
           outcomes_text: str, date_str: str) -> str:
    """One LLM call: review recent outcomes against the current lessons and
    rewrite them as one short block. Non-fatal on model failure."""
    p = Path(memory_path)
    # A pre-archive file is the old free-form log (invented process rules the
    # agent cannot act on); start the new block clean rather than anchor on it.
    prior = "" if p.exists() and not archive_path(p).exists() else read_memory(p)
    prompt = (
        "You are reviewing your own recent paper-trading decisions.\n"
        "Each day, for each symbol, you see only last_close, momentum_5d, "
        "vol_20d and current_pos, and you answer -1 (short), 0 (flat) or "
        "1 (long). That is everything you control.\n\n"
        f"Current lessons:\n{prior or '(none yet)'}\n\n"
        f"Recent decisions and outcomes:\n{outcomes_text or '(no decisions to review)'}\n\n"
        f"Rewrite the lessons as at most {MAX_BULLETS} bullets. Each bullet is one "
        "line starting with '- ', at most 30 words, and is a rule you can apply "
        "with only those inputs and that answer. Cite a number from the "
        "outcomes above for each. Keep a current lesson only if the outcomes "
        "still support it. No process rules (checklists, audits, sign-offs, "
        "IDs, caps, schedules) -- you cannot act on them. If the outcomes "
        "support no rule, give one bullet saying so. Output only the bullets."
    )
    try:
        text = complete(prompt).strip()
    except Exception as e:
        # Loud but non-fatal: a silently-skipped reflection looks identical to
        # a deliberate no-op in run.json, which defeats the audit trail.
        print(f"!! reflect: model call failed: {e}", file=sys.stderr)
        return ""
    if not text:
        print("!! reflect: model returned empty text", file=sys.stderr)
        return ""
    bullets = _bullets(text)
    if not bullets:
        print("!! reflect: model reply had no bullet lines -- lessons kept as-is",
              file=sys.stderr)
        return ""
    write_lessons(p, date_str, bullets)
    return bullets
