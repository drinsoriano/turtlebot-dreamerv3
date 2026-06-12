# Training Convergence — How Long Is Long Enough?

A practical reference for choosing the training budget (`--steps`). The short
version: **train only until the metric you care about plateaus.** Past that point,
extra steps are mostly wasted compute — they do not improve success rate, and
path efficiency improves through **reward shaping**, not more steps. The plateau
point is **stage-dependent**.

---

## 1. TL;DR

- **Success rate** plateaus near the ceiling **very early** (~15–35% of a 40k-step
  stage-1 run). The rest of training adds essentially nothing to it.
- **Path efficiency** plateaus **lower (~0.70–0.75) and noisily** — it wobbles
  rather than climbing, and more steps do not push it toward 1.0.
- So "longer = better" is true only **up to convergence**. After that it is
  diminishing returns.
- Convergence is **stage-dependent**: easy stages (1–2) converge fast; cluttered
  stages (3+) need more steps.

---

## 2. Evidence — stage-1 BO trials (~38k episodes)

**Success rate** (rolling-100) at run quartiles — already maxed by 25%:

| trial | episodes | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| 000 | 1213 | 98.0% | 99.0% | 99.0% | 100% |
| 001 | 1293 | 100% | 100% | 100% | 100% |
| 002 | 1458 | 100% | 100% | 100% | 100% |
| 003 | 921 | 94.0% | 100% | 99.0% | 100% |

**Path efficiency** (rolling-100 of `planner_status=="ok"` rows) — plateaus
low and is non-monotonic (note the ↓):

| trial | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| 001 | 0.679 | 0.736 | 0.727 | 0.712 ↓ |
| 002 | 0.699 | 0.713 | 0.751 | 0.705 ↓ |
| 005 | 0.636 | 0.649 | 0.679 | 0.758 |

Convergence detection (the dashboard's `_convergence_point`, tol = 2 pts for
success, 0.03 for efficiency) on the same trials:

| trial | success converges @ | excess budget | efficiency |
|---|---|---|---|
| 000 | ep 279 (23% of run) | **77%** | settles ~0.70, only near the end (noisy) |
| 001 | ep 225 (18%) | **82%** | ~0.715 |
| 002 | ep 220 (15%) | **85%** | ~0.757 |

> **Reading it:** success is done learning after ~1/5 of the run; the remaining
> ~80% is "excess" for that metric. Efficiency, by contrast, never cleanly
> plateaus on stage 1 — it keeps wandering within a noisy band.

---

## 3. Diminishing returns vs. cost

On this machine a 40k-step trial takes ~140 min, so:

| `--steps` | Wall-clock | Success | Efficiency |
|---|---|---|---|
| 40,000 | ~140 min | ~100% | ~0.72 |
| 300,000 | ~17.5 h | ~100% | ~0.72 |
| 600,000 | ~35 h | ~100% | ~0.72 |

**35 hours for the same result as ~2.3 hours** on stage 1. The extra steps buy
almost nothing.

---

## 4. Efficiency comes from reward shaping, not more steps

The agent reaches ~100% success but stays at ~0.72 efficiency no matter how long
it trains — it learns **goal-reaching** without learning **shortest-path**
behavior (see the *Metric Interpretation* note in `CLAUDE.md` and
[methodology.md](methodology.md)). To push efficiency higher, change the **reward**
(reward shaping / BO weight tuning), not the step count. That is exactly what the
`shaped` reward mode and `tune_reward.py` are for.

---

## 5. Choosing `--steps` per stage

| Stage | Character | Suggested budget |
|---|---|---|
| 1–2 | nearly obstacle-free, converges fast | 40k–100k is often enough to *rank* configs; 300k for final reported numbers |
| 3–5 | cluttered, slower convergence | 300k standard; watch the curve, extend if still rising |
| 6+ | large / hardest | ≥300k; verify the plateau before trusting results |

> **Default trap:** with `--configs turtle` and **no** `--steps`, the run uses
> **600,000** steps ([configs.yaml:108](../dreamerv3-torch/configs.yaml#L108),
> which overrides the `defaults:` value of 300k at
> [configs.yaml:10](../dreamerv3-torch/configs.yaml#L10)). That is usually longer
> than needed — **pass `--steps` explicitly** instead of relying on the default.

The BO proxy budget (`--steps 40000`) only *ranks* candidate weights; the winner
is then *confirmed* at full budget (see the validation protocol in `CLAUDE.md`).

---

## 6. Measure it yourself — the 🎯 Convergence dashboard tab

```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch/dashboard
streamlit run app.py
```

Open **🎯 Convergence**, pick a run and a metric (Success Rate or Path
Efficiency). The tab draws the rolling curve, a dotted **plateau line**, and a
**★ marker** at the convergence episode, plus a summary table with
`pct_of_run` and **`excess_pct`** (share of episodes that ran *after*
convergence). A high `excess_pct` means you could safely train with fewer
`--steps` on that stage. Adjust the **tolerance** slider to see how strict the
"plateau" definition is — for the noisy efficiency curve a looser tolerance finds
an earlier settle point.

---

## 7. Buod (Tagalog)

- **Mas mahaba ≠ mas maganda** kapag **naka-plateau na** ang metric. Sa Stage 1,
  ang success rate ay ~100% na agad (~15–35% ng run lang) — sayang na ang natira.
- Ang **path efficiency** ay nananatili sa ~0.70–0.75 at **maingay** — hindi ito
  umaakyat kahit gaano kahaba ang training.
- Para tumaas ang efficiency: **baguhin ang reward** (reward shaping / BO tuning),
  **hindi** ang `--steps`.
- **Pumili ng `--steps` ayon sa stage:** maiksi para sa Stage 1–2, mas mahaba para
  sa Stage 3+. Kapag walang `--steps`, **600k** ang default sa `turtle` config —
  kadalasang sobra, kaya maglagay ng `--steps` palagi.
- Gamitin ang **🎯 Convergence** tab para makita kung saan nag-plateau at kung
  gaano karaming steps ang sobra (`excess_pct`).
