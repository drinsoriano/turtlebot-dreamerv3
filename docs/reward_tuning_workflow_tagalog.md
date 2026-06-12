# Reward Tuning at Validation — Tagalog na Reference

Gabay (sa Tagalog) kung paano gamitin ang Bayesian Optimization (BO / Optuna)
para sa reward shaping, at paano i-validate ang panalong weights sa stages 1–8.
Isinulat ito para maiwasan ang ilang karaniwang pagkakamali sa interpretasyon.
Para sa teknikal na detalye sa Ingles, tingnan ang
[methodology.md](methodology.md) at ang CLAUDE.md (seksyon **Reward-Weight
Tuning (Bayesian Optimization)**).

> **Pinakamahalagang punto:** Ang **reward weights** (5 `--reward_*` numbers) ay
> **input/setting** na ikaw ang naglalagay — **hindi** ito output ng training. Ang
> training ay gumagawa ng **model (`.pt`)** at **metrics**, hindi ng reward weights.
> Kaya ang weights ay **pinipili sa BO search lang** — hindi sa validation runs.

---

## Ang 5 reward weights

Ang shaped reward ay may 5 na ti-tune na timbang (`reward_mode: shaped`):

| Weight | Default | Sakop ng BO search |
|---|---|---|
| `reward_progress_scale` | 1.0 | 0.1 – 3.0 (log) |
| `reward_step_penalty` | 0.01 | 0.0 – 0.05 |
| `reward_turn_penalty` | 0.01 | 0.0 – 0.05 |
| `reward_near_obstacle_scale` | 0.1 | 0.0 – 0.5 |
| `reward_near_obstacle_sigma` | 0.25 | 0.1 – 0.5 |

Isang **set** ng 5 numbers na ito ang tinatawag na "config" o "weights."

---

## Ang 3 phase ng buong proseso

### Phase 1 — BO Search (tuning)
- `tune_reward.py` ay nagpapatakbo ng **30 trials** (default).
- **Bawat trial = ISANG candidate set ng 5 weights**, sinasanay sa **maikling
  budget** (`--steps 40000`, 1 seed lang).
- I-si-score ang bawat trial gamit ang eval `planner_path_efficiency`, na may
  **constraint**: ang success rate ay hindi dapat bumagsak nang higit sa `margin`
  (default 5 puntos) mula sa baseline.
- **Output:** **ISANG** best feasible config (5 numbers).
- 👉 **Dito lang pumipili ng weights.** 30 candidates → 1 panalo.

### Phase 2 — Validation (3 seeds, stage 1)
- Kunin ang **ISANG** panalong config mula sa Phase 1.
- Patakbuhin ito nang **3 beses** sa **buong budget** (`--steps 300000`,
  `--eval_episode_num 100`), seed `0`, `1`, `2` — **parehong 5 weights** sa lahat.
- **Walang bagong weights na ginagawa dito.** Ang seeds ay para lang patunayan na
  **matatag** ang config (hindi swerte ng isang seed).
- **Output:** 3 set ng **metrics** (success, efficiency). **I-average ang metrics**
  (mean ± std) para sa report — **hindi** ang weights.

### Phase 3 — Transfer test (stages 2–4)
- Gamitin ang **parehong ISANG** config ng 5 weights.
- **Fresh training** kada stage (bagong logdir, random init, walang kinakargang
  `.pt`).
- Sinusukat: nag-tra-**transfer** ba ang reward *recipe* sa mas magulong arena?

---

## Mga karaniwang MALING akala (linawin)

| Maling akala | Tama |
|---|---|
| "Ang seed0–seed2 ay gagawa ng 90 weights." | ❌ Hindi. Ang validation ay nagpapatakbo ng **panalong config lang (1) × 3 seeds = 3 runs**. Walang bagong weights. Ang "30 × 3" ay hindi ang validation. |
| "Pipili ako ng best weights mula sa 3 seeds." | ❌ Hindi. Ang weights ay **galing na sa BO** (1 set). Ang training run ay hindi gumagawa ng weights. |
| "I-a-average ko ang weights ng 3 seeds." | ❌ Hindi. **Iisa** lang ang weights. Ang ina-average ay ang **metrics** (success, efficiency). |
| "I-transfer ko ang best `.pt` sa stage 2 → 3 → 4." | ❌ Hindi. Ang dinadala ay ang **5 reward numbers**, hindi ang `.pt`. Fresh training kada stage. |
| "Curriculum learning ito." | ❌ Hindi. Walang ipinapasang model sa susunod na stage (tingnan sa baba). |

### Bakit HINDI ito curriculum learning

| | **Validation (ito ang ginagawa)** | **Curriculum learning (iba ito)** |
|---|---|---|
| Dala sa susunod na stage | 5 reward **numbers** lang | Ang buong **`.pt` model** |
| Simula ng stage 2/3/4 | **From scratch** (random init) | Ipinagpapatuloy ang nakaraang model |
| Tanong na sinasagot | Nag-tra-transfer ba ang reward *recipe*? | Mas mabilis bang matuto kung paunti-unti ang hirap? |

Sa codebase, ang resume ay nag-lo-load ng `latest.pt` **sa parehong logdir lang**.
Kung bagong logdir ang stage 2, **from scratch** talaga — walang cross-stage
`.pt` transfer by default.

---

## Analohiya: recipe

- **Reward weights** = recipe (dami ng 5 sangkap).
- **BO (Phase 1)** = nagluto ng **30 recipe**, tinikman, **pinili ang
  pinakamasarap** (1 recipe). ← dito pumili.
- **Validation 3 seeds (Phase 2)** = niluto ang **panalong recipe** ng 3 beses para
  siguraduhing **palaging masarap** (hindi swerte). Ina-average ang "rating"
  (metrics), **hindi** ang recipe.
- **Stage 2–4 (Phase 3)** = niluto ang **parehong recipe** sa ibang kusina (mas
  mahirap na arena).

---

## Paano kunin ang best weights (pagkatapos ng 30 trials)

Pumipili ang BO ng best **feasible** trial (pinakamataas na efficiency, basta
hindi bumagsak ang success nang higit sa margin).

**Opsyon A — `export_tune_results.py` (CSV sa `csv_logs/tune_stage{N}/`):**
```bash
cd ~/turtlebot-dreamerv3/dreamerv3-torch
python3 export_tune_results.py --stage 1
```
Sumusulat ng:
- `csv_logs/tune_stage1/tune_trials_stage1.csv` — **lahat** ng trials (best at
  hindi best) + 5 weights kada isa.
- `csv_logs/tune_stage1/tune_best_stage1.csv` — ang **best feasible** lang (1 row).

**Opsyon B — direktang basahin ang study DB:**
```bash
python3 - <<'PY'
import optuna
s = optuna.load_study(study_name="reward_stage1",
                      storage="sqlite:///tune_reward_stage1.db")
done = [t for t in s.trials if t.value is not None]
feas = [t for t in done if t.user_attrs.get("constraint",(1.0,))[0] <= 0]
best = max(feas or done, key=lambda t: t.value)
print("best trial", best.number, "efficiency", round(best.value,4),
      "success", best.user_attrs.get("success"))
for k,v in best.params.items(): print(f"  --{k} {v:.6g}")
PY
```

> **Babala:** Huwag i-re-run ang `tune_reward.py --n-trials 30` para lang makita ang
> best — magdadagdag iyon ng 30 PANG trials. Gamitin ang Opsyon A o B.

---

## Validation command (full budget)

Ipasok ang **iisang** best 5 weights. Para sa **stage 1**, ulitin sa seed `0,1,2`;
para sa **transfer**, palitan ang `--stage` at `--logdir` (2, 3, 4) — **parehong
5 weights**:

```bash
python3 dreamer.py --configs turtle --task turtle \
  --logdir ./logdir/stage1_360_none_seed0_reward_tuned \
  --stage 1 --lidar 360 --odometry_mode none --seed 0 \
  --device cuda --steps 300000 --eval_episode_num 100 \
  --reward_mode shaped \
  --reward_progress_scale <v> --reward_step_penalty <v> --reward_turn_penalty <v> \
  --reward_near_obstacle_scale <v> --reward_near_obstacle_sigma <v>
```

Para sa report: kunin ang **mean ± std** ng `success_rate` at
`planner_path_efficiency` mula sa `blackbox_eval_*` at `planning_eval_*` CSVs ng
3 seeds. (Tingnan ang [whitebox_data_validation.md](whitebox_data_validation.md):
gamitin ang **`blackbox_eval`** para sa success/collision, hindi ang whitebox.)

---

## Buod (cheat sheet)

| Bagay | Ilan | Inaverage? | Saan nanggaling |
|---|---|---|---|
| Reward weights (5 numbers) | **1 set** | ❌ Hindi | BO search (Phase 1) |
| Metrics ng 3 seeds (success, eff) | 3 | ✅ Oo → mean ± std | Validation runs (Phase 2) |
| Weights na dadalhin sa stage 2–4 | **1 set** (pareho) | ❌ Hindi | Galing pa rin sa BO |
| `.pt` checkpoint | 1 kada run | — | Output ng bawat training; **hindi** dinadala sa susunod |

**Daloy:** 30 BO trials → **1 best config** → 3-seed validation sa stage 1
(average ang **metrics**) → kung maganda, dalhin ang **parehong 5 numbers** sa
stage 2–4 (fresh training) para sa transfer test.
