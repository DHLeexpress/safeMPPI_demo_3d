# 0814 multi-sphere bowling route handoff

민혁에게 넘길 **fixed-γ bowling trajectory package**입니다. 공개 사이트의 기존
`PRE2`, `S4 raw`, `S4 distinct` 뷰는 그대로 보존했고, 아래 다섯 paper-ready
view를 모두 **γ=0.1**로 맞췄습니다.

| View | 실제 rollout 수 | 고정 γ | 의미 |
|---|---:|---:|---|
| `paper-ready-pre2` | 8 | 0.1 | PRE2 raw flow, NFE16·M1 |
| `paper-ready-less-expanded` | 8 | 0.1 | speed400 expansion Round 1, NFE16·M1 |
| `paper-ready-expanded` | 8 | 0.1 | expanded S4 raw flow, NFE16·M1 |
| `paper-ready-cfmmppi` | **8 × 3 regimes = 24** | 0.1 | PRE2-based CFM–MPPI safety / balanced / performance |
| `paper-ready-safemppi` | 8 | 0.1 | exact published SafeMPPI controller |

사이트에서 CFM regime 하나를 선택한 상태에는 요청대로 **5 Views × 8 = 40**
trajectories가 대응합니다. Frozen bundle은 CFM 세 regime을 모두 보존하므로 실제
저장량은 PRE2 8 + R1 8 + Expanded 8 + CFM 24 + SafeMPPI 8 = **56**입니다.

별도로 `mirrored_cylinder_gamma03/`에는 PRE2의 실제 pretraining law였던
randomized vertical-cylinder scene에서 고른 γ=0.3 reference 8개가 있습니다.
네 개의 exact axis-180 source/mirror pair로 구성되어 있으며, start-goal 축 기준
`LLL`부터 `RRR`까지 8개 lateral signature를 하나씩 담았습니다. 사이트 View는
`paper-ready-pre2-mirrored-cylinder-gamma0p3`입니다. 이 묶음은 서로 다른 random
scene의 simulation reference이므로 bowling용 56개 flight roster에는 합산하지
않습니다.

## Flight references — position, velocity, acceleration at 100 Hz

`flight_references/` contains a ready-to-stream reference for every one of the
56 frozen paper trajectories. Minhyuk must use these files instead of
differentiating the displayed positions:

```text
time_s             (T,)    float64, uniformly spaced at 0.01 s
position_ref       (T, 3)  float32, stored dense rollout positions
velocity_ref       (T, 3)  float32, exact once-governed recurrence
acceleration_ref   (T, 3)  float32, applied 10 Hz control held at 100 Hz
```

`applied_controls_10hz` and the 0806-compatible alias
`executed_controls_10hz` are both included and are byte-identical. The player
must not reapply the governor, interpolate, smooth, or differentiate position.
The authoritative lookup table is `flight_references/FLIGHT_INDEX.csv`, and
`flight_references/manifest.json` records the source row and SHA-256 for every
reference.

The 40-trajectory visible comparison is obtained by choosing one CFM regime.
The package exports all three CFM regimes, so the reference count is 56:

| method/view | reference count | hardware status |
|---|---:|---|
| PRE2 | 8 | simulated successes; operator approval required |
| Less-expanded R1 | 8 | simulated successes; operator approval required |
| Expanded S4 | 8 | simulated successes; operator approval required |
| CFM–MPPI safety | 8 | 5 success / 3 known collision |
| CFM–MPPI balanced | 8 | 5 success / 3 known collision |
| CFM–MPPI performance | 8 | 8 known collision |
| SafeMPPI | 8 | simulated successes; operator approval required |

Every non-successful CFM reference is marked `SIMULATION_ONLY_KNOWN_COLLISION`
in the index and must never be sent to hardware. Successful references still
require the normal operator and hardware safety approval.

Validation-only playback example:

```bash
python minhyuk/frozen_reference_player.py \
  --reference flight_references/expanded/gamma_0p1_e227_seed_100399_100hz.npz
```

The player validates the arrays and sends no command by itself. Connect its
`send_full_state(position, velocity, acceleration)` callback only inside the
versioned hardware runner. To regenerate the references from the frozen
handoff without running any planner or policy:

```bash
python source/export_flight_references.py --bundle . --output /tmp/0814_refs
```

`SITE_TRAJECTORY_INDEX.csv`가 View → method/regime/γ/episode/rollout seed/status/
route/source row를 잇는 authoritative index입니다. 사이트의 focus selector와
trajectory card에도 같은 seed가 표시됩니다.

## 1. 바로 사용할 trajectory와 seed

### PRE2 — γ=0.1, 8 trajectories

모두 faithful raw NFE16·M1 rollout입니다. 모델은
`checkpoints/pre2/pretrained.pt`입니다.

| episode | seed | route |
|---:|---:|:---:|
| 15 | 92555 | RRR |
| 22 | 92814 | RRL |
| 0 | 92000 | LLL |
| 19 | 92703 | LLL |
| 14 | 92518 | LLL |
| 7 | 92259 | LLL |
| 12 | 92444 | LLL |
| 21 | 92777 | LLL |

PRE2는 Expanded와 동일한 discovery trial budget에서 최대 3개 mode를
보여주도록 선택했습니다. 따라서 일부러 8-mode 균형으로 꾸민 결과가 아닙니다.

### Less-expanded — speed400 Round 1, γ=0.1, 8 trajectories

정확한 model은 `checkpoints/less_expanded/checkpoint_001.pt`입니다. γ=0.1의
50 trials 중 15개가 success였고, positive goal progress + hard z-band를 통과한
후 4 modes가 남았습니다. Route balance를 먼저 적용한 뒤 mode 안에서 quality
score가 좋은 순서로 8개를 선택했습니다.

| episode | seed | route |
|---:|---:|:---:|
| 29 | 93073 | LLL |
| 46 | 93702 | LLR |
| 47 | 93739 | LRL |
| 48 | 93776 | LRR |
| 24 | 92888 | LLL |
| 4 | 92148 | LRL |
| 26 | 92962 | LLL |
| 17 | 92629 | LLL |

Checkpoint SHA-256은
`7d5c5cc1f7e9b55ae2a803b8a3812264f38072cd74a416cd81b32b9810253f98`,
expansion source snapshot ID는 `5c8a57779f16-1cf210b2e2ee`입니다. 정확한
snapshot은 `source_snapshots/less_expanded_r1/`, original raw 50×4 bowling
bank의 R1-only copy는
`trajectories/less_expanded/r1_only_bowling_raw_trajectories.pt`에 있습니다.
Bowling rollout을 만든 evaluation snapshot은 별도로
`5c8a57779f16-5ab0794c77dd`이며
`source_snapshots/less_expanded_r1_evaluation/`에 그대로 보존했습니다.
핵심 source hashes는 runner `d4a7a3f8…`, expansion loop `98af2265…`,
multi-pair task adapter `f22398cc…`이며 full 값은 `bundle_manifest.json`과
`provenance/less_expanded/SOURCE_IDS.json`, `SHA256SUMS`에 있습니다.

### Expanded S4 — γ=0.1, 8 trajectories

모델은 `checkpoints/expanded/checkpoint_004.pt`입니다. 각 route code마다 한
trajectory를 선택했습니다.

| episode | seed | route |
|---:|---:|:---:|
| 227 | 100399 | LLL |
| 230 | 100510 | LLR |
| 260 | 101620 | LRL |
| 297 | 102989 | LRR |
| 121 | 96477 | RLL |
| 267 | 101879 | RLR |
| 4 | 92148 | RRL |
| 60 | 94220 | RRR |

### CFM–MPPI — γ=0.1, **8 trajectories per regime**

여기서 “8개”는 전체 8개가 아니라 아래 세 종류가 각각 같은 8개 seed를
rollout한 것입니다. 따라서 사이트와 handoff에는 총 24개가 있습니다.

| regime | normalized reward | normalized safety | 결과 |
|---|---:|---:|---|
| `safety` | 0.0 | 1.0 | 5 success / 3 collision |
| `balanced` | 0.5 | 1.0 | 5 success / 3 collision |
| `performance` | 1.0 | 0.0 | 0 success / 8 collision |

공통 seed는 `314159, 314196, 314233, 314270, 314307, 314344, 314381,
314418`입니다. 이 세 regime은 서로 다른 controller setting이므로 같은 seed의
세 결과도 서로 다른 trajectory입니다. CFM–MPPI는 PRE2를 generative prior로
사용하며, NFE16, proposal 32, top-8 elites, elite당 Gaussian copy 32,
σ=0.2, λ=0.1, α=0.5, CBF margin=0.1 m입니다. H_P/NVP verifier와 progress
label은 proposal selection에 사용하지 않습니다.

### SafeMPPI — γ=0.1, 8 trajectories

| episode | seed | route | quality penalty ↓ |
|---:|---:|:---:|---:|
| 832 | 245332 | LLL | 0.155098 |
| 628 | 245128 | LLR | 0.436818 |
| 18 | 240018 | RRL | 0.403625 |
| 1623 | 246523 | RRR | 0.426758 |
| 1622 | 246522 | LLL | 0.188764 |
| 1349 | 246249 | LLR | 0.565285 |
| 1416 | 246316 | RRL | 0.571209 |
| 1055 | 245555 | RRR | 0.427094 |

SafeMPPI는 learned checkpoint가 없는 controller baseline입니다. `seed`는
controller rollout RNG seed이고, exact source는 `paper_ready/0808/safemppi`
및 이 번들의 pinned `runtime_snapshot/safe_mppi`입니다. 기존 100회와 추가
search를 합친 γ=0.1 총 1,700 trials에서, hard 조건을 통과한 전체 후보를 같은
route-balance/quality score 기준으로 다시 정렬해 최종 8개를 선택했습니다.
노출된 modes는 LLL/LLR/RRL/RRR 각 2개입니다. 표의 score는 낮을수록 좋습니다.

## 2. Frozen arrays를 바로 읽는 법

```python
import torch

payload = torch.load(
    "trajectories/paper_ready_bowling_handoff.pt",
    map_location="cpu",
    weights_only=False,
)

pre2 = payload["groups"]["paper-ready-pre2"]               # 8
expanded = payload["groups"]["paper-ready-expanded"]       # 8
less_expanded = payload["groups"]["paper-ready-less-expanded"] # 8
cfmmppi = payload["groups"]["paper-ready-cfmmppi"]          # 24 = 8 × 3
safemppi = [
    row for row in payload["groups"]["paper-ready-safemppi"]
    if abs(float(row["gamma"]) - 0.1) < 1e-9
]                                                               # 8
```

각 row에는 적어도 `states`, `controls`, `applied_controls`, `gamma`,
`episode`, `rollout_seed`, `status`, `bowling_route`가 있습니다. CFM row에는
`regime`, `normalized_goal`, `normalized_safety`도 있습니다. SafeMPPI의 dense
path 필드는 `dense_positions`, 나머지 method는 `dense_steps`입니다.

먼저 integrity를 검사하십시오.

```bash
python VERIFY.py
```

`SHA256SUMS`가 모든 frozen file을 묶고, `VERIFY.py`는 다음을 추가로 검사합니다.

- PRE2 8 / Less-expanded R1 8 / Expanded 8 / SafeMPPI 8이 모두 γ=0.1인지
- CFM–MPPI safety, balanced, performance가 각각 8개이고 모두 γ=0.1인지
- Expanded R1, CFM–MPPI, SafeMPPI site array가 원본 raw bank와 exact-equal인지
- 사이트에 모든 selected seed가 노출되는지

## 3. Published seed를 실제 모델/controller로 재실행

기록된 backend와 같은 accelerator family를 사용하십시오. 아래는 번들 root에서
실행하는 명령입니다.

```bash
DEVICE=cuda:0 OUT=/tmp/0814_reproduction bash REPRODUCE.sh
```

이 명령은 frozen PRE2/Expanded seed를 `--verify-frozen`으로 재실행하여
state/control/dense-step mismatch가 있으면 중단하고, CFM–MPPI fixed γ=0.1
matched bank와 selected SafeMPPI seeds도 다시 생성합니다. 정확한 GPU
floating-point byte identity는 기록된 runtime/device에 묶입니다. 다른 backend의
결과는 scientific replication이지 byte identity 보장은 아닙니다.

## 4. 민혁이 seed를 바꾸어 보는 법

### PRE2 또는 Expanded

`source/reproduce_policy_site_rollouts.py`의 `--seeds`는 frozen verification을
끄고 임의 seed를 실제 model로 rollout합니다.

```bash
python source/reproduce_policy_site_rollouts.py \
  --device cuda:0 --model pre2 --gamma 0.1 \
  --seeds 12345 12346 12347 \
  --output /tmp/pre2_custom.pt

python source/reproduce_policy_site_rollouts.py \
  --device cuda:0 --model expanded --gamma 0.1 \
  --seeds 22345 22346 22347 \
  --output /tmp/expanded_custom.pt

python source/reproduce_policy_site_rollouts.py \
  --device cuda:0 --model less-expanded --gamma 0.1 \
  --seeds 32345 32346 32347 \
  --output /tmp/less_expanded_custom.pt
```

두 경우 모두 site와 같은 faithful raw NFE16·M1, temperature 1.0,
E15+wall250+axis5+control0.05+speed400 deployment입니다. verifier 또는 progress
label로 action을 고르지 않습니다.

### CFM–MPPI

runner에서 base seed를 주면 실제 rollout seed는 `base + 37*trial`입니다.

```bash
python source/run_multisphere_cfm_mppi_bowling.py \
  --pretrain-dir checkpoints/pre2 \
  --task-config config/task_config_resolved.json \
  --output /tmp/cfmmppi_custom \
  --device cuda:0 --gammas 0.1 --trials 8 --seed 500000 \
  --proposal-count 32 --elite-count 8 --copies-per-elite 32 \
  --mppi-sigma 0.20 --mppi-lambda 0.10 \
  --alpha-cbf 0.5 --cbf-margin-m 0.10 \
  --goal-coefficient-max 0.25 --safety-coefficient-max 1.0 \
  --regimes-json '{"safety":{"goal":0.0,"safety":1.0},"balanced":{"goal":0.5,"safety":1.0},"performance":{"goal":1.0,"safety":0.0}}'
```

세 regime 모두를 반드시 구분해서 보고하십시오. `w_safe=w_goal=0`이어도 MPPI
refinement가 남으므로 raw PRE2와 같지 않습니다. 세부 수식과 coefficient
normalization은 `CFM_MPPI_METHOD_CONTRACT.md`가 source of truth입니다.

### SafeMPPI

새 seed 하나를 시도하려면 `--seed-start`에 그 값을 넣습니다.

```bash
python source/collect_paper_ready_safemppi_bowling.py \
  --source-root runtime_snapshot \
  --config config/safemppi_exact_bowling_config.json \
  --output /tmp/safemppi_seed_600000 \
  --device cuda:0 --gammas 0.1 \
  --attempts-per-gamma 1 --seed-start 600000
```

연속 N개 seed는 `--attempts-per-gamma N`으로 생성합니다. 실제 seeds는
`seed-start, ..., seed-start+N-1`입니다. exact controller SHA가 다르면 wrapper가
실행 전에 fail-close합니다.

## 5. Method/source provenance

| role | artifact | SHA-256 |
|---|---|---|
| PRE2 policy | `checkpoints/pre2/pretrained.pt` | `76b10a69b6f26d65533d4e617ccbf6fb77a2178ac030244c5a745c11a4d0a3c0` |
| less-expanded R1 | `checkpoints/less_expanded/checkpoint_001.pt` | `7d5c5cc1f7e9b55ae2a803b8a3812264f38072cd74a416cd81b32b9810253f98` |
| expanded S4 policy | `checkpoints/expanded/checkpoint_004.pt` | `a37f93091bfc88340b7f2bab0d41cb889ff59a2caf00d737e58e57fbfa4cdb52` |
| PRE2/Expanded scene | `config/task_config_resolved.json` | `baf3a8f3398cba147696ee783957b865968186e905279d1300a87477459792fc` |
| SafeMPPI scene | `config/safemppi_exact_bowling_config.json` | `33474ab44f637ed1177f05f2ad86848555f263dcf33ef4a802b92250b0511ef1` |
| SafeMPPI controller | `runtime_snapshot/safe_mppi/controller.py` | `dfc91a26ccac2818c902215bf4d9a06e405d5878e5c6af0be2f75c4f68106dad` |

SafeMPPI published source commit은
`dabb5011dfc674864e1de275a1e1c2adab58f4af`입니다. 이미 공개된 controller
handoff는 [`paper_ready/0808/safemppi`](../0808/safemppi/README.md), 공통
provenance는 [`paper_ready/common/PROVENANCE.md`](../common/PROVENANCE.md)에
있습니다. 0814 bundle에는 site를 그대로 재생성하는 데 필요한 runtime snapshot,
models, configs, raw banks, selected full arrays, seed index, renderer source가 모두
들어 있습니다.

## 6. Site 재생성과 view 보존

`source/build_paper_ready_bowling_handoff.py`가 raw banks에서 selection JSON,
full-array handoff, inner/outer sandbox HTML을 재생성합니다.
`source/paper-ready-bowling-handoff.html`이 renderer template입니다. 기존 PRE2
한 개와 S4 두 개 view는 새 paper-ready view와 별도로 보존됩니다.

`not-paper-ready-*`에는 승격 γ를 제외한 다른 gamma bank가 남아 있습니다.
CFM–MPPI의 과거 paper γ=0.3도 삭제하지 않고 `not-paper-ready-cfmmppi`로
이동했습니다. 따라서 paper fixed-γ comparison과 exploratory gamma 결과를
혼동하지 마십시오.

공개 사이트는 `bundle_manifest.json`의 Sites project ID와 URL에 고정되어 있고,
`site/visualization.html`은 그 배포의 exact sandboxed iframe artifact입니다.
