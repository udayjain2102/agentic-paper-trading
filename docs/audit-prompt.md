# Skeptical audit prompt

Paste this into a fresh session at the repo root.

---

You are auditing this repository (`robinhood agentic trading`) with maximum skepticism. Assume nothing in the README, docstrings, comments, or commit messages is true until you have verified it by reading code or running it. Your job is NOT to fix anything. Your job is to tell me what this actually is, what actually runs, and what is broken, fake, or dead.

Ground every claim in evidence: a file path with line numbers, a command you ran and its output, or a git fact. If you cannot verify something, say "unverified" — do not fill the gap with a plausible story.

Work through these, in order:

**1. What is this thing supposed to do?**
State the intended purpose in three sentences, sourced from the code, not the docs. Then quote where the docs disagree with the code.

**2. What is the real entry point?**
Trace the actual execution path end to end: what invokes what, from CLI/cron/CI trigger down to the trade decision and its side effects. Draw the flow. Name every file that is actually on that path.

**3. What is dead?**
List every module, script, config key, and data file that nothing on the live path reaches. For each, say how you determined it is unreachable. Include: files nothing imports, config values nothing reads, tests that test nothing real, CI jobs that never fire or always pass vacuously.

**4. Does it run at all?**
Actually try. Run the tests. Run the main entry point in whatever the safest mode is. Report the real output including failures. If it cannot run, say precisely why — missing env var, missing credential, missing data, import error. Do not describe what would happen if it worked.

**5. Where is the data really coming from?**
For every source of prices, fundamentals, or signals: is it live, cached, stale, synthetic, or hardcoded? Check timestamps on the cached files. Check whether fallback chains silently substitute fake data for real data. Flag anywhere a failure degrades to a made-up number instead of an error.

**6. Are the results real?**
Look at the journals, backtest outputs, and P&L numbers. Are they produced by the code as it currently stands, or are they stale artifacts from an older version? Check for lookahead bias, survivorship bias, missing transaction costs, missing slippage, in-sample fitting, and any place a number is computed from data that would not have been available at that point in time. Be specific about which output files are trustworthy and which are not.

**7. What is load-bearing and undefended?**
Which single points of failure would silently produce wrong results rather than crash? Where would a bug be invisible?

**8. What is over-built?**
Which abstractions, layers, config options, or agent scaffolding exist without earning their keep — one implementation behind an interface, config that never varies, retries around things that do not fail, an LLM call doing what ten lines of Python would do?

**9. The questions I should be asking.**
End with the five questions about this project I have not asked and should have — the ones where you suspect the answer is uncomfortable.

Format: short sections, blunt language, no hedging, no praise. Lead with a one-paragraph verdict: does this system work, and if it appears to work, is that appearance real?
