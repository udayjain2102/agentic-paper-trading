# Safety drills

Evidence that the kill switches actually fire in the real `paper_run` path, not
only in unit tests. Re-run these before any change to `guardrails.py`,
`executor.py`, `paper_account.py`, or `paper_run.py`, and before `LIVE=true` is
ever considered.

All drills run against the real `config.yaml` limits and the real cached bars in
`data/`, writing state to a scratch directory so the tracked record is untouched.

## 2026-07-26

Config at time of drill: `per_trade_max_usd=250`, `total_deployed_max_usd=2000`,
`max_new_positions_per_run=2`, `max_orders_per_run=5`, `daily_loss_limit_usd=200`.

### D1 — normal run, guardrails bind

```
12 orders proposed, 2 accepted, 10 rejected
first rejection: DIS | "Max new positions per run reached (2)."
first fill:      CMCSA $250 filled
account after:   cash 5000 -> 4500, 2 positions with shares + cost basis
```

The caps bind. Before the persistent account existed they could not: `Account`
was rebuilt empty every run, so `total_deployed_max_usd`,
`max_new_positions_per_run` and `daily_loss_limit_usd` compared against nothing.

### D2 — HALT file stops trading

```
touch HALT   -> {'halted': True, 'orders': 0, 'accepted': 0}
                reason: "HALT file present -- trading halted by operator."
rm HALT      -> {'halted': False, 'orders': 10, 'accepted': 2}
```

### D3 — daily loss limit fires

Seeded `realized_pnl_today_usd = -250.0` against a `-200` limit, dated today.

```
{'halted': True, 'orders': 0, 'accepted': 0}
reason: "Daily loss limit hit: realized P&L $-250.00 <= -$200.00."
```

### D4 — daily loss limit resets on a new trading day

Same `-250.0`, but dated the previous day. A limit that fires and never resets
would brick trading permanently after one bad session, so this half matters as
much as D3.

```
{'halted': False, 'orders': 11, 'accepted': 2}
realized_pnl_today_usd rolled to 0.0, date advanced
positions preserved across the roll
```

### D5 — state persists across runs

Run 1 proposed 12 orders. Run 2, reading the state run 1 wrote, proposed 10 —
the two positions bought in run 1 were held and not re-bought.

## Not yet drilled

- The daily-loss switch firing from *organically accumulated* losses rather than
  a seeded value. D3 proves the check works; it does not prove `apply_fill`
  books realized losses correctly enough to reach the limit on its own.
- Anything against a real broker. `paper_run` is MockBroker-only by assertion.
- Recovery from a corrupted or partially-written `paper_account.json`.
