# Graph Report - .  (2026-07-15)

## Corpus Check
- 122 files · ~72,830 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 964 nodes · 2090 edges · 43 communities (32 shown, 11 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 426 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- LLM Decision Agent
- Decision Engine Adapters
- Dashboard Generation
- Decision Seam & Overlays
- Cross-Sectional IC Math
- Backtest & Comparison
- Strategy Base Contract
- OOS Split & Gate
- Safety & Overlay Concepts
- Backtest Selection Concepts
- Strategy Search Loop
- Price Data Fetch & Cache
- Strategy-Mode Order Runner
- Entry Features
- Paper-Trade Loop Tests
- Multiple-Testing Stats Concepts
- OOS Gate Statistics
- Paper-Trade Engine Concepts
- IC & Search Concepts
- v1 Bot Safety & Runner
- Backtest Infrastructure
- Engine Adapter Design
- OOS Gate Implementation
- Strategy Implementations
- Quant Loop Sub-Projects
- Factor CLI Tests
- Paper-Trade CLI Tests
- Paper Cron Script
- Gate CLI Tests
- Search CLI Tests
- Headless Refresh Test
- Order Execution Seam
- Factor Package Init
- Gate Package Init
- Search Package Init
- Backtest Metrics
- Loss-Learning Bake-Off
- rhagent Package

## God Nodes (most connected - your core abstractions)
1. `Decision` - 31 edges
2. `RunState` - 30 edges
3. `build()` - 29 edges
4. `Order` - 27 edges
5. `PaperTrader` - 27 edges
6. `OrderExecutor` - 26 edges
7. `StrategyEngine` - 24 edges
8. `MockBroker` - 23 edges
9. `Gates` - 22 edges
10. `AgentEngine` - 21 edges

## Surprising Connections (you probably didn't know these)
- `test_decision_is_frozen()` --calls--> `Decision`  [INFERRED]
  tests/test_engine.py → src/rhagent/engine.py
- `test_verdict_reasons_in_order()` --calls--> `verdict()`  [INFERRED]
  tests/gate/test_oos.py → src/rhagent/gate/oos.py
- `test_build_overlay_none()` --calls--> `build_overlay()`  [INFERRED]
  tests/test_overlay_seam.py → src/rhagent/overlay.py
- `test_close_fill_fills_at_close()` --calls--> `CloseFill`  [INFERRED]
  tests/test_papertrade_helpers.py → src/rhagent/papertrade.py
- `_bakeoff_table()` --calls--> `robust_table()`  [INFERRED]
  scripts/make_dashboard.py → src/rhagent/evaluate_robust.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Offline improvement flow (loops A-E)** — architecture_loop_a_search, architecture_loop_b_oos_gate, architecture_loop_c_papertrade, architecture_loop_d_overlays, architecture_loop_e_forward_record [EXTRACTED 0.90]
- **Durable weekday paper-run pipeline** — github_workflows_daily_paper_run_workflow, scripts_paper_cron_paper_cron_sh, architecture_loop_e_forward_record, github_workflows_daily_paper_run_paper_state_branch, architecture_refresh_cache_first [EXTRACTED 0.85]
- **Shared safety funnel** — architecture_llm_decision_agent, architecture_rule_based_strategies, architecture_order_executor, architecture_safety_funnel [EXTRACTED 0.90]
- **Quant loop: IC evaluation → search → OOS gate** — docs_superpowers_plans_2026_07_12_cross_sectional_ic_evaluator_rank_ic, docs_superpowers_plans_2026_07_12_strategy_search_loop_search_result, docs_superpowers_plans_2026_07_12_oos_gate_strict_verdict [EXTRACTED 0.85]
- **Paper-trade overlays bake-off** — docs_superpowers_plans_2026_07_13_loss_learning_phase1_overlay_protocol, docs_superpowers_plans_2026_07_13_loss_learning_phase1_conviction_gate, docs_superpowers_plans_2026_07_13_loss_learning_phase1_bucket_filter, docs_superpowers_plans_2026_07_14_loss_learning_phase2_winprob_winprob_gate [EXTRACTED 0.85]
- **Strategies → backtest engine → comparison ranking** — docs_superpowers_plans_2026_07_06_backtest_strategy_selection_strategy_base, docs_superpowers_plans_2026_07_06_backtest_strategy_selection_backtest_engine, docs_superpowers_plans_2026_07_06_backtest_strategy_selection_compare_cli [EXTRACTED 0.85]
- **Quant strategy-search framework (IC evaluator -> search loop -> OOS gate)** — docs_superpowers_specs_2026_07_12_cross_sectional_ic_evaluator_design_ic_evaluator, docs_superpowers_specs_2026_07_12_strategy_search_loop_design_search_loop, docs_superpowers_specs_2026_07_12_oos_gate_design_oos_gate [EXTRACTED 1.00]
- **Order safety funnel (LLM/strategy -> place_order wrapper -> guardrails -> broker MCP)** — docs_superpowers_specs_2026_06_16_robinhood_agentic_trading_design_place_order_wrapper, docs_superpowers_specs_2026_06_16_robinhood_agentic_trading_design_guardrails, docs_superpowers_specs_2026_06_16_robinhood_agentic_trading_design_broker [EXTRACTED 1.00]
- **Paper-trade eval pipeline (DecisionEngine -> papertrade engine -> trade ledger -> evaluate)** — docs_superpowers_specs_2026_07_11_papertrade_eval_loop_design_decision_engine, docs_superpowers_specs_2026_07_11_papertrade_eval_loop_design_papertrade_engine, docs_superpowers_specs_2026_07_11_papertrade_eval_loop_design_trade_ledger, docs_superpowers_specs_2026_07_11_papertrade_eval_loop_design_evaluate [EXTRACTED 1.00]

## Communities (43 total, 11 thin omitted)

### Community 0 - "LLM Decision Agent"
Cohesion: 0.05
Nodes (87): OpenAI, _dispatch(), Any, The decision agent: an LLM reasons over the portfolio and proposes trades.  The, A no-API stand-in for ``run_session``.      Walks the same tool-dispatch path th, Run the agent loop for one cron tick. Returns the model's final text., run_scripted_session(), run_session() (+79 more)

### Community 1 - "Decision Engine Adapters"
Cohesion: 0.05
Nodes (58): AgentEngine, DecisionEngine, DataFrame, Protocol, Adapt a vectorized Strategy: the last value of positions(history) is     the tar, Let an LLM pick today's position. Same DecisionEngine protocol as     StrategyEn, StrategyEngine, build_overlay() (+50 more)

### Community 2 - "Dashboard Generation"
Cohesion: 0.07
Nodes (64): _bakeoff_table(), _buckets_table(), _compare_table(), _equity_svg(), _latest_run(), main(), _money(), _num() (+56 more)

### Community 3 - "Decision Seam & Overlays"
Cohesion: 0.06
Nodes (50): Decision, Decision, The decision seam between the paper-trade loop and whatever decides.  A Decision, apply_conviction(), BucketFilter, ConvictionGate, _fit_logit(), _predict_logit() (+42 more)

### Community 4 - "Cross-Sectional IC Math"
Cohesion: 0.07
Nodes (52): DatetimeIndex, forward_returns(), half_life(), ic_decay(), ic_series(), icir(), DataFrame, Series (+44 more)

### Community 5 - "Backtest & Comparison"
Cohesion: 0.06
Nodes (52): BacktestResult, net_returns(), DataFrame, Series, Offline backtest engine.  Turns a target-position series into a net-return serie, result_from_returns(), run(), _aggregate() (+44 more)

### Community 6 - "Strategy Base Contract"
Cohesion: 0.07
Nodes (33): clamp_short(), DataFrame, Series, The common contract every strategy implements.  A strategy maps a DataFrame of d, Long-only guard: map short signals (-1) to flat (0) unless shorting is on., Today's target position (last row only). Default recomputes the whole         se, Continuous score aligned to bars.index; higher = more bullish on the         for, Strategy (+25 more)

### Community 7 - "OOS Split & Gate"
Cohesion: 0.07
Nodes (42): in_sample_mask(), oos_cutoff(), Series, The locked in-sample / out-of-sample date boundary.  The out-of-sample slice is, load_universe(), DataFrame, GateResult, GateRow (+34 more)

### Community 8 - "Safety & Overlay Concepts"
Cohesion: 0.06
Nodes (44): BucketFilter overlay, clamp_short / long-only default, Conviction gate overlay, Daily-loss kill switch, Dry-run by default (LIVE=true), Fully-realized-day guard, linreg strategy, LLM decision agent (agent.py) (+36 more)

### Community 9 - "Backtest Selection Concepts"
Cohesion: 0.05
Nodes (44): Backtest engine, Backtest & Strategy Selection, clamp_short (long-only guard), Strategy comparison CLI, Data fetch + CSV cache, LinReg strategy, MeanReversion strategy, Momentum strategy (+36 more)

### Community 10 - "Strategy Search Loop"
Cohesion: 0.12
Nodes (39): apply_gates(), config_key(), first_failing_gate(), Gates, _half_life_ok(), The search loop: four survival gates and the coarse-to-fine round loop.  Gates (, RoundLog, run_search() (+31 more)

### Community 11 - "Price Data Fetch & Cache"
Cohesion: 0.10
Nodes (31): Pull the structured/JSON payload out of an MCP CallToolResult., _structured(), get_bars(), mcp_fetch(), _normalize(), DataFrame, Path, Historical price data: fetch from the Robinhood MCP, cache to CSV.  Cache-first: (+23 more)

### Community 12 - "Strategy-Mode Order Runner"
Cohesion: 0.12
Nodes (24): Trade the configured winning strategy through the executor/guardrails., run_strategy_mode(), Momentum, DataFrame, Series, Pairs, pairs_target_orders(), DataFrame (+16 more)

### Community 13 - "Entry Features"
Cohesion: 0.15
Nodes (19): datetime, entry_features(), flatten_trades(), DataFrame, Lookahead-free entry-time features, shared by the ledger writer and overlays., Flatten a trades frame's nested `entry_features` dict column into     `feat_*` c, Cheap lookahead-free scalars at entry, used for failure bucketing., new_run_id() (+11 more)

### Community 14 - "Paper-Trade Loop Tests"
Cohesion: 0.18
Nodes (16): _bars(), FakeSource, Emits a fixed target sequence per symbol — fully deterministic., _run(), ScriptedEngine, test_cost_bps_charged_on_round_trip(), test_determinism_same_inputs_same_ledger(), test_diverging_symbol_indices_raise() (+8 more)

### Community 15 - "Multiple-Testing Stats Concepts"
Cohesion: 0.13
Nodes (18): Bailey & Lopez de Prado, Bonferroni correction, Deflated Sharpe Ratio, evaluate.py failure buckets, IC decay / half-life, ICIR (IC Information Ratio), Cross-sectional Information Coefficient (IC), The locked OOS split (factor/split.py) (+10 more)

### Community 16 - "OOS Gate Statistics"
Cohesion: 0.19
Nodes (13): Locked out-of-sample split, Bailey & Lopez de Prado, Bonferroni correction, Deflated Sharpe Ratio, n_eff overlapping-window discount, Normal CDF / PPF (Acklam), Out-of-Sample Gate, Strict viability verdict (+5 more)

### Community 17 - "Paper-Trade Engine Concepts"
Cohesion: 0.17
Nodes (13): journal.py (JSONL audit trail), No-lookahead invariant, Determinism guarantee, evaluate.py (ledger, stats, failure buckets, run compare), Failure buckets (entry_features), MarketSource / FillModel seams, papertrade.py event-driven engine, Trade ledger + run_id/trade_id scheme (+5 more)

### Community 18 - "IC & Search Concepts"
Cohesion: 0.22
Nodes (9): Rank-IC not beta-neutral caveat, Information Coefficient (IC), Rank-IC / ICIR / decay (ic.py), ConfigScore / score_config (score.py), Four survival gates (ICIR/half-life/robustness/sign), n_tested (distinct configs count), In-sample search overfits by design, run_search round loop + SearchResult (+1 more)

### Community 19 - "v1 Bot Safety & Runner"
Cohesion: 0.25
Nodes (8): Robinhood Agentic Trading Bot v1, Dry-run-by-default safety posture, guardrails.py, Daily realized-loss kill-switch + HALT file, place_order wrapper, runner.py, Backtest & Strategy Selection, Runner strategy mode integration

### Community 20 - "Backtest Infrastructure"
Cohesion: 0.25
Nodes (8): broker.py, Robinhood Trading MCP Server, backtest.py vectorized engine, compare.py ranking & selection, data.py OHLCV fetch + CSV cache, Total-return ranking metric, Paper-Trading & Evaluation Loop, universe.py (~60-name large-cap panel)

### Community 21 - "Engine Adapter Design"
Cohesion: 0.29
Nodes (7): Claude Agent SDK, AgentEngine adapter (later increment), DecisionEngine interface (engine.py), StrategyEngine adapter, ConvictionGate overlay, Decision.conviction field, ParamTune / TunedStrategyEngine

### Community 22 - "OOS Gate Implementation"
Cohesion: 0.33
Nodes (7): Bonferroni-adjusted p-value, Deflated Sharpe Ratio (Lopez de Prado), evaluate_oos + strict verdict (oos.py), run_gate orchestration + GateResult, stats.py (norm cdf/ppf, pure no-scipy), Strict both-corrections viability verdict, evaluate_robust.py (fold/bootstrap/deflated Sharpe)

### Community 23 - "Strategy Implementations"
Cohesion: 0.40
Nodes (6): Linear-regression strategy, Mean-reversion strategy (z-score), Momentum / trend strategy, Pairs trading strategy, Strategy interface (base.py), Continuous signal() contract

### Community 24 - "Quant Loop Sub-Projects"
Cohesion: 0.40
Nodes (5): Cross-Sectional IC/ICIR Evaluator (factor/), split.py locked OOS split, Out-of-Sample Gate + Multiple-Testing Correction (gate/), Expect zero-viable on thin data (honest outcome), Strategy Search Loop (search/)

### Community 25 - "Factor CLI Tests"
Cohesion: 0.60
Nodes (3): _seed(), test_cli_empty_ic_series_reports_insufficient_data(), test_cli_reports_icir()

### Community 26 - "Paper-Trade CLI Tests"
Cohesion: 0.60
Nodes (3): _seed_cache(), test_cli_compare_lists_runs(), test_cli_runs_and_writes_ledger()

## Knowledge Gaps
- **64 isolated node(s):** `rhagent`, `paper_cron.sh script`, `PYTHONPATH`, `Weekday cron schedule (11:17 UTC)`, `scripts/paper_cron.sh` (+59 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build()` connect `Cross-Sectional IC Math` to `LLM Decision Agent`, `Decision Engine Adapters`, `Dashboard Generation`, `Decision Seam & Overlays`, `Backtest & Comparison`, `Strategy Base Contract`, `Strategy-Mode Order Runner`?**
  _High betweenness centrality (0.200) - this node is a cross-community bridge._
- **Why does `Strategy` connect `Strategy Base Contract` to `Decision Engine Adapters`, `Decision Seam & Overlays`, `Cross-Sectional IC Math`, `Strategy-Mode Order Runner`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `load()` connect `LLM Decision Agent` to `Decision Seam & Overlays`, `Dashboard Generation`, `Price Data Fetch & Cache`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `Decision` (e.g. with `BucketFilter` and `ConvictionGate`) actually correct?**
  _`Decision` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `RunState` (e.g. with `ExecuteResult` and `OrderExecutor`) actually correct?**
  _`RunState` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `build()` (e.g. with `test_signal_panel_shape_and_alignment()` and `test_linreg_signal_matches_position_sign_where_in_position()`) actually correct?**
  _`build()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `Order` (e.g. with `Broker` and `Fill`) actually correct?**
  _`Order` has 17 INFERRED edges - model-reasoned connections that need verification._