# White-box Data Validation

A read-only audit of the **black-box** and **white-box** CSV logging in the
current DreamerV3 navigation project, answering one question: *are the white-box
data correct, usable, and consistent with the black-box data?*

No training was run and no source code was modified for this audit. Findings are
grounded in the logging code and in the CSV files present on disk under
`dreamerv3-torch/csv_logs/`. Confidence labels follow the rest of `/docs`:
**implemented** (direct code/data evidence), **inferred from implementation**,
**not found**.

> **Headline verdict.** The white-box *learning* diagnostics (losses, KL,
> entropies, returns) are **correct and usable**. Two columns —
> `eval_success_rate` and `eval_collision_rate` — were **broken in shaped-reward
> runs** (logged as `0.0` even when the agent was succeeding); this was fixed in
> code on 2026-06-09, but **existing pre-fix CSVs still contain the wrong values**.
> White-box and black-box files for tuner runs were previously **split across
> folders**; the white-box CSV now **honours `csv_dir`** (fixed 2026-06-09) and is
> written alongside the other per-run CSVs. For success/collision rates, use the
> **black-box eval** CSV, not white-box.

---

## 1. Where black-box CSV logging is implemented

**Implemented.** `dreamerv3-torch/envs/turtle.py`, in the environment.

- Routing and header: `Env.init_properties` opens `blackbox_{run}.csv` (train) or
  `blackbox_eval_{run}.csv` (eval) under `csv_dir` ([turtle.py:155-193](../dreamerv3-torch/envs/turtle.py#L155-L193)).
- Row written per terminal episode by `_write_blackbox_csv(outcome)`
  ([turtle.py:384](../dreamerv3-torch/envs/turtle.py#L384)), called from the three
  terminal branches of `get_reward_and_done` (success / collision / timeout).
- **Columns (15):** `datetime, odometry_mode, stage, episode, outcome,
  steps_to_goal, path_directness, min_obstacle_dist, near_collisions,
  success_rate, collision_rate, rolling_success_rate_100,
  rolling_collision_rate_100, rolling_success_rate_500, rolling_collision_rate_500`.

## 2. Where white-box CSV logging is implemented

**Implemented.** `dreamerv3-torch/tools.py`, in the `Logger`.

- Header opened in `Logger.__init__` ([tools.py:82-93](../dreamerv3-torch/tools.py#L82-L93)).
- Row written in `Logger.write()` **only when `model_loss` is present in the
  scalar dict** ([tools.py:141](../dreamerv3-torch/tools.py#L141)).
- **Columns (23):** the original 13 — `datetime, step, train_return,
  reward_variance, model_loss, actor_loss, value_loss, kl, prior_ent, post_ent,
  eval_return, eval_success_rate, eval_collision_rate` — **plus 10 diagnostics
  appended 2026-06-12:** `reward_variance_100, actor_entropy, model_grad_norm,
  actor_grad_norm, value_grad_norm, reward_loss, dyn_loss, rep_loss, ema_005,
  ema_095`.
- **2026-06-12 diagnostics.** All ten were **already computed** in `models.py`
  every training step and flushed through the logger; they were simply not
  persisted to the CSV. Persisting them is **zero extra training cost** and they
  are **appended at the end of the row** so the original 13 columns keep their
  positions (pre-2026-06-12 files just lack the new columns — read by name).
  - `actor_entropy` — policy entropy; collapse toward ~0 = exploration stopped
    (looping / stuck). The most useful health signal.
  - `model_grad_norm` / `actor_grad_norm` / `value_grad_norm` — gradient L2 norms;
    spikes flag instability.
  - `reward_loss` — isolated reward-head loss (high ⇒ reward unpredictable ⇒
    planning degrades). `dyn_loss` / `rep_loss` — the KL split (posterior-collapse
    check).
  - `ema_005` / `ema_095` — return-normalization percentiles (reward-scale drift).
  - `reward_variance_100` — rolling last-100 reward variance. The original
    `reward_variance` is **cumulative (all-time)** and stays high forever because
    early-training chaos is never dropped; the rolling version is the honest
    "stable *now*?" signal. Both are retained.
- **Path honours `csv_dir`** (fixed 2026-06-09): `{csv_dir}/whitebox_{run}.csv`
  ([tools.py:82](../dreamerv3-torch/tools.py#L82)), co-located with the other
  per-run CSVs (see §8). Previously it was hardcoded to `./csv_logs/`.

## 3. Black-box vs white-box — the difference

| | Black-box | White-box |
|---|---|---|
| Question answered | *What did the robot do?* | *How is the algorithm learning?* |
| Writer | Environment (`turtle.py`) | `Logger` (`tools.py`) |
| Granularity | **Per episode** | **Per logging interval** (steps) |
| Example fields | outcome, steps_to_goal, path_directness, success/collision rates | model/actor/value loss, KL, prior/post entropy, train/eval return |
| Identity columns | `odometry_mode`, `stage` present | none — only `step` + metrics |

Black-box is **behavioural** (externally observable navigation outcomes);
white-box is **algorithm-internal** (the learning signals of the world model and
actor-critic). They are complementary: black-box says whether navigation improved,
white-box says whether the model is training stably.

## 4. Is black-box per-episode and white-box interval-based?

**Confirmed (implemented).**

- Black-box: one row per terminal episode (`_write_blackbox_csv` is called from the
  done branches only).
- White-box: one row per logging interval. On disk, white-box `step` advances in
  **exact increments of 2000** (e.g. `500, 2500, 4500, …`), matching the turtle
  `log_every` (see §5).

## 5. White-box logging trigger

**Implemented.** Two conditions must both hold for a white-box row:

1. The agent's periodic log fires: `if self._should_log(step)` every `log_every`
   steps ([dreamer.py:71-78](../dreamerv3-torch/dreamer.py#L71-L78)), which dumps
   the training metrics (incl. `model_loss`) into the logger and calls
   `logger.write()`.
2. `Logger.write()` emits a row only when `model_loss` is in the scalar dict
   ([tools.py:141](../dreamerv3-torch/tools.py#L141)).

`log_every` is `2e3` for the `turtle` config (`configs.yaml` l.109) and is divided
by `action_repeat` (=1) in `dreamer.py` l.185 → **effective interval = 2000 steps**.
The per-episode `logger.write()` call inside `simulate()`
([tools.py:300](../dreamerv3-torch/tools.py#L300)) does **not** carry `model_loss`,
so it writes **no** white-box row — confirming the cadence is the 2000-step log
interval, not the episode.

## 6. Should white-box have fewer rows than black-box?

**Yes — confirmed (implemented).** White-box is interval-based, black-box is
per-episode, and many episodes complete within one 2000-step interval. Observed on
disk:

| Run | White-box rows | Black-box train rows | Black-box eval rows |
|---|---|---|---|
| `reward_stage1_baseline_seed0` | 23 | 1189 | 60 |
| `reward_stage1_trial000` | 23 | 1213 | 60 |
| `reward_stage1_trial001` | 23 | 1293 | 60 |
| `reward_stage1_trial002` | 23 | 1458 | 60 |
| `reward_stage1_trial003` | **0** | 2 | 14 |

White-box ≈ 1–2 % of black-box train rows, as expected. The `trial003` case is a
warning sign (see §9): the run was so short it never reached the first 2000-step
log, so its white-box file has a header but **zero data rows** while black-box
still has episodes.

## 7. White-box CSV column-level checks

Checks run over the white-box files in `csv_logs/`:

| Check | Result |
|---|---|
| correct `run_name` | **Only via filename** (`whitebox_{run}.csv`). There is **no `run_name` column.** Filenames are correct and match the logdir basename. |
| correct `stage` | **No `stage` column exists** in white-box. Not recoverable from the file. (Black-box has it.) |
| correct `seed` | **No `seed` column exists** in white-box *or* black-box. Seed is only implied by `run_name` *if* the run name encodes it (`…_seed0`); the tuner runs (`…_trial000`) do **not** encode the seed at all. |
| correct `odometry_mode` | **No `odometry_mode` column exists** in white-box. (Black-box has it; all inspected runs are `none`.) |
| increasing `step` | **Pass.** Monotonic increasing, no duplicates; uniform 2000-step spacing (`500 → 44500`). |
| no NaN | **Pass.** No NaN in any numeric column of any non-empty file. |
| no infinity | **Pass.** No ±inf in any file. |
| non-constant diagnostics | **Pass for learning diagnostics.** e.g. `model_loss` 32.8 → 1.44, `kl` 0.85 → 2.77, `train_return` varies per row. **Fail for the eval-rate columns in shaped runs** — see §9. |

**Key structural finding:** the white-box CSV has **no identity columns**
(`run_name`, `stage`, `seed`, `odometry_mode`). Its only key is the filename. Any
cross-referencing to stage / seed / odometry mode must go through the matching
**black-box** file or the logdir name.

## 8. Are black-box and white-box aligned for the same run?

**By `run_name` (filename), and now co-located in the same `csv_dir`.**

- There is **no shared key inside the files** (white-box lacks `stage`, `seed`,
  `odometry_mode`). Alignment relies entirely on the `{run_name}` in the filename
  matching between `whitebox_{run}.csv` and `blackbox_{run}.csv`.
- **Folder location (fixed 2026-06-09).** The white-box CSV now honours `csv_dir`,
  so for the **tuner runs** both families are written to the same directory:
  - white-box → `csv_logs/tune_stage1/whitebox_reward_stage1_trial000.csv`
  - black-box → `csv_logs/tune_stage1/blackbox_reward_stage1_trial000.csv`

  The dashboard's folder picker therefore shows white-box **and** black-box for the
  same selected folder. (Previously white-box was pinned to `./csv_logs/` while the
  tuner's other CSVs went to `tune_stage1/`; **existing pre-fix files remain where
  they were written** and are not moved automatically.)

**Consequence:** alignment is now folder-consistent. Older runs created before the
fix keep their white-box CSV in `./csv_logs/`; new and resumed runs write all
per-run CSVs (black-box, white-box, planning, reward, resource) into the same
`csv_dir`.

## 9. Suspicious signs

- 🔴 **Broken eval rates in shaped-reward runs (pre-fix data).** In
  `whitebox_reward_stage1_trial000/001/002.csv` (shaped mode),
  `eval_success_rate` and `eval_collision_rate` are **`0.0` on every row**, even
  though `eval_return ≈ 99–100`. The matching black-box eval shows the truth — for
  `trial000`: 47 success / 9 timeout / 4 collision, cumulative success 78.33 %,
  collision 6.67 %. Cause: the old code scored eval outcomes by comparing episode
  reward to literal `100`/`-10`, which never matches once shaping terms are added.
  **Fixed in code on 2026-06-09** (now scored from the terminal outcome label), but
  **the existing pre-fix CSVs above remain wrong.**
- 🟠 **Timeout counted as collision (default mode, pre-fix).** Because timeout also
  returns `-10`, the old eval `collision_rate` folded timeouts in. The
  `baseline_seed0` file (default mode) shows non-zero rates but its
  `eval_collision_rate` is really *(collision + timeout)*. Black-box eval keeps the
  three outcomes separate. Also fixed 2026-06-09.
- 🟠 **Empty white-box file for short runs.** `trial003` has 0 data rows (run
  stopped before the first 2000-step log) while black-box has episodes — a
  white-box/black-box row-count mismatch that is expected but can look like data
  loss.
- 🟡 **Forward-filled eval columns.** `eval_return`, `eval_success_rate`,
  `eval_collision_rate` repeat across rows between evaluations (e.g. `eval_return`
  shows only 3 distinct values across 23 rows). The actual eval points are correct;
  a naive plot must de-duplicate or the eval curve looks like a stair-step.
- 🟡 **No reset-step counters within a file.** Each run's white-box `step` is clean
  and monotonic; no mid-file resets were found. Different runs reuse the same step
  range (`500–44500`) in their own files — expected, not a defect.
- ✅ **`csv_dir` now honoured** (fixed 2026-06-09; see §8) — white-box is written
  in the same folder as the other per-run CSVs. Only files created *before* the fix
  remain in the main `csv_logs/` folder.
- ✅ **No mixed modes / no duplicate rows / no missing columns** within any single
  file: each black-box file is single-stage, single-odometry; white-box headers are
  complete and column order is stable.

## What the white-box data represent

The white-box CSV is the **algorithm-internal training record** of the DreamerV3
agent, sampled every 2000 steps:

- `model_loss` — world-model training loss (reconstruction + reward + continue + KL).
- `actor_loss`, `value_loss` — actor-critic objectives over imagined rollouts.
- `kl`, `prior_ent`, `post_ent` — latent KL divergence and prior/posterior
  entropies of the RSSM.
- `train_return` — the most recent completed training-episode return at that step.
- `reward_variance` — cumulative variance of training-episode returns since process
  start (a global figure, not a rolling window).
- `eval_return`, `eval_success_rate`, `eval_collision_rate` — last evaluation
  summary, forward-filled until the next evaluation.

## How they differ from black-box metrics

Black-box describes **behaviour per episode** and carries identity columns
(`stage`, `odometry_mode`); white-box describes **learning dynamics over step
intervals** and carries no identity columns. Black-box is the authoritative source
for success rate, collision rate, path quality, and per-episode outcomes;
white-box is the authoritative source for losses, KL, entropies, and learning
trends.

## How to know if white-box data are correct

1. **`step` monotonic, uniform spacing, no dups** → logging cadence intact.
2. **No NaN / inf** in numeric columns → numerically healthy training.
3. **`model_loss` trends down and `kl` stays bounded** → world model is learning,
   not diverging.
4. **`train_return` / `eval_return` rise toward the success reward (~100)** →
   policy improving.
5. **Cross-check eval rates against black-box eval.** If `eval_success_rate` is
   `0` while `eval_return ≈ 100`, the row is from the **pre-fix bug** — trust
   `blackbox_eval_{run}.csv` instead.
6. **Confirm the matching black-box file exists for the same `run_name`** (possibly
   in a `tune_stage{N}/` subfolder) to recover `stage` / `odometry_mode`.

## What checks were performed (this audit)

- Located both writers in code (§1, §2) and the trigger (§5).
- Parsed all five `whitebox_*.csv` files and their black-box counterparts.
- Verified step monotonicity/spacing, NaN/inf, constant-column, and row-count
  ratios (§6, §7).
- Compared white-box eval columns against black-box eval outcomes (§9).
- Confirmed the folder split between white-box and tuner black-box (§8).

## Issues found (summary)

| Severity | Issue | Status |
|---|---|---|
| 🔴 High | `eval_success_rate` / `eval_collision_rate` = 0 in shaped runs | Code fixed 2026-06-09; pre-fix CSVs still wrong |
| 🟠 Med | Timeout counted as collision in eval rates (default mode) | Code fixed 2026-06-09 |
| ✅ Resolved | White-box `csv_dir` ignored → split from black-box | Fixed 2026-06-09 (now honours `csv_dir`) |
| 🟡 Low | No identity columns (`run_name`/`stage`/`seed`/`odometry_mode`) | By design; rely on filename + black-box |
| 🟡 Low | Forward-filled eval columns; empty file for ultra-short runs; cumulative `reward_variance` | By design; interpret accordingly |

## How white-box data should be used in methodology and results

- **Use white-box for the learning story.** Report `model_loss`, `actor_loss`,
  `value_loss`, `kl`, `prior_ent`/`post_ent`, and `train_return` vs step as
  evidence of **training stability and convergence** of the DreamerV3 world model
  and actor-critic. These are the trustworthy, NaN/inf-free, monotonically-stepped
  diagnostics.
- **Do not report success/collision rate from white-box.** Use
  `blackbox_eval_{run}.csv` (`outcome`, `success_rate`, `collision_rate`) — it is
  correct in both reward modes and separates timeout from collision. Treat the
  white-box `eval_success_rate`/`eval_collision_rate` only as a quick monitor on
  **post-fix** runs, and disregard them entirely on the pre-fix tuner CSVs.
- **For `eval_return`, de-duplicate** to the rows where evaluation actually ran
  (values change), or read eval return from the evaluation episodes directly.
- **State the cadence.** White-box is sampled every 2000 steps; black-box is
  per-episode — so white-box has far fewer rows by design, and the two are aligned
  only by `run_name`.
- **Record the run identity externally** (logdir name / a run table), because the
  white-box file itself does not encode `stage`, `seed`, or `odometry_mode`.

> Cross-references: metric definitions and the train/eval routing are in
> [training_and_evaluation_pipeline.md](training_and_evaluation_pipeline.md); the
> eval-rate fix is described in the project notes for 2026-06-09; the overall
> evidence base is [implementation_audit.md](implementation_audit.md).
