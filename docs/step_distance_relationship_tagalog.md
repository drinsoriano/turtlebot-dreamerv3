# Step ↔ Oras ↔ Layo — Ano Talaga ang "Step" (Tagalog)

Gabay (sa Tagalog) para maintindihan ang ugnayan ng training **step**,
simulated na **oras**, **layo** na natatakbo ng robot, at **isang ikot ng gulong**
(wheel rotation). Madaling mapagkamalan ang mga ito kaya sobrang layo ng mali sa
bilang. Para sa kompletong tsart/constant tables at empirical data sa English,
tingnan ang [step_distance_relationship.md](step_distance_relationship.md).

> **Pinakamahalaga:** ang training ay binibilang sa **steps** (ito ang
> convention sa reinforcement learning). Ang **metro** ay *derived* lang — at
> **hindi** ito laging pareho kada step. Ang training budget = **steps**; ang
> navigation quality = **metro**.

---

## Ang isang maling akala na kailangang itama agad — ang step ay ORAS, hindi LAYO

> Ang **step** ay isang *tik ng desisyon* — isang **fixed na piraso ng oras** kung
> saan gumagalaw ang robot sa **bilis na pinili niya**. **Hindi** ito fixed na layo.

Ang bitag: iniisip mong ang step ay "isang fixed na hakbang/layo na nilakad ng
robot." Ang tama: **isang sandali kung saan nagdesisyon ang robot.** Ang layo ay
**resulta** ng desisyong iyon, hindi ang kahulugan ng step.

Nag-iiba ang bilis **dahil ang bilis MISMO ang aksyon**
([turtle.py:850](../dreamerv3-torch/envs/turtle.py#L850)):

```python
linear_vel  = abs(action[0]) * 0.1     # ito ang PINIPILI ng robot kada step
angular_vel = action[1] * 2 * 0.1
```

| Pinili ng robot | linear_vel | layo sa isang step (~1 segundo) |
|---|---|---|
| `action[0] = 1.0` (bilis na bilis) | 0.10 m/s | ~0.10 m |
| `action[0] = 0.5` (katamtaman) | 0.05 m/s | ~0.05 m |
| `action[0] = 0.0` (hinto / umikot lang) | 0.00 m/s | ~0 m |

**Parehong step, parehong tagal — pero iba't ibang layo**, dahil iba't iba ang
bilis na pinili. Kung hindi puwedeng mag-iba ang bilis, **hindi makakahinto ang
robot sa goal at hindi makakabagal pag malapit sa obstacle.** Ito mismo ang
**natututunan** ng DreamerV3 — kung kailan bibilis, kailan babagal.

---

## Mga analohiya (sa Tagalog)

**1. Metronome ("tik... tik... tik...").**
Ang isang step ay parang **isang "tik"** ng metronome. Steady lang ang tunog —
pantay-pantay ang **oras** sa pagitan ng mga tik. Pero kung ilang **metro** ang
malalakad mo sa pagitan ng dalawang tik — depende kung **tumatakbo ka o
naglalakad lang**. Ang "tik" ay sukat ng **oras**, hindi ng layo.

**2. Sasakyan at gas pedal (kada 1 segundo).**
Isipin mong **kada 1 segundo** ay tinitingnan mo ang manibela at pedal. Yung "1
segundo" na iyon = isang step (fixed na oras). Pero kung ilang **metro** ang
naabot mo sa loob ng 1 segundo — depende kung **gaano mo kalakas tinapakan ang
gas**. Buong tapak → malayo. Inalis ang paa → halos hindi gumalaw. **Pareho ang
oras, iba ang layo.**

**3. Laro / video game (60 FPS).**
Bawat "frame" ay parehong dami ng **oras**. Pero kung gaano kalayo gumalaw ang
karakter mo sa isang frame — depende kung **gaano mo kalakas idiniin ang
joystick**. Ang frame ay "kailan ka magde-desisyon", hindi "gaano kalayo ka
gumalaw".

---

## Ang apat na "orasan" — iisa lang ang FIXED na conversion

| Conversion | Fixed ba? | Bakit |
|---|---|---|
| ikot ng gulong ↔ **metro** | ✅ fixed | circumference ng gulong (0.207 m) — purong geometry |
| **step** ↔ metro | ❌ nag-iiba | layo = piniling bilis × sim-`dt`; pinipili ang bilis kada step |
| step ↔ ikot ng gulong | ❌ nag-iiba | sunod sa pagka-iba ng step↔metro |
| step ↔ sim-oras | ≈ humigit-kumulang | ~0.6–1.4 s, depende sa `spin_once`/RTF |

**Ang ikot ng gulong ay wala sa control loop.** Ang policy ay nag-uutos ng
`/cmd_vel` (linear + angular **velocity**). Ang differential-drive plugin ng
Gazebo ang nag-cocompute ng bilis ng gulong (`v = ω_gulong × r`, `r = 0.033 m`).
Sa 0.1 m/s na cap: ω ≈ 3.0 rad/s ≈ 0.48 ikot/segundo ≈ isang ikot kada ~2.1 s.
Hindi kailanman inuutos o nakikita ng robot ang ikot ng gulong — *resulta* lang
ito, isang antas sa ilalim ng `cmd_vel`.

- **1 ikot ng gulong = π × 0.066 ≈ 0.207 m** (fixed; walang slip). Ito ang
  **tanging** conversion na talagang constant.

---

## Buod (cheat sheet)

| Tanong | Tamang yunit |
|---|---|
| Gaano karaming compute ang ginamit sa training? | **steps** |
| Maganda ba/efficient ang daan na tinahak? | **metro** (`actual_path_length`, A\* path) |
| Gaano kabilis nakarating sa goal? | steps-to-goal (≈ segundo, dahil halos constant ang `dt`) |

- ❌ *"Step = fixed na layo na nilakad"* → **MALI**.
- ✅ *"Step = isang tik ng oras kung saan gumagalaw ang robot sa piniling bilis"* → **TAMA**.
- Nag-iiba ang bilis dahil **ang bilis ang aksyon**. Kaya: **steps** ang training
  budget (parang orasan, steady); **metro** ang layo (resulta ng piniling bilis).

> **Empirical (galing sa data, hindi haka-haka):** ~0.065 m kada step (umaabot ng
> ~0.14 m); isang episode (≤250 steps) ay umaabot ng **~26 m** kapag timeout.
> Tingnan ang [step_distance_relationship.md](step_distance_relationship.md) para
> sa kumpletong tables, constants, at ang dahilan kung bakit MALI ang "2.5 m"
> na hula. Para sa `planner_path_efficiency` (dimensionless m/m, kaya kanselado
> ang step↔metro factor), tingnan din ang [methodology.md](methodology.md).
