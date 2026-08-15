# 0815 real bowling flight handoff

민혁이가 승인된 공개 사이트의 **real-paper-ready 5 views × 8 trajectories =
40 trajectories**를 그대로 확인하고 flight reference로 사용할 수 있는 handoff입니다.

- Published view: <https://s4-monotone-bowling-routes.leedo12663.chatgpt.site/>
- Fixed conditioning: `gamma = 0.3`
- Measured obstacle centers/radii: `scene/bowling_as_built.csv`
- Hard modeled obstacle: each measured physical radius **+ 0.16 m**
- Hanging-string no-go cylinder: each physical ball top에서 ceiling까지, radius **0.10 m**
- Reference rate: 100 Hz
- Full source control rate: 10 Hz

## 1. 먼저 이것만 실행

Repository root에서:

```bash
python paper_ready/0815/VERIFY.py
```

정상 결과:

```text
OK: ... 5 views, 40 exact 100 Hz references,
38 hardware-eligible, 2 simulation-only
```

`flight_references/FLIGHT_INDEX.csv`에서 원하는 row를 고른 뒤 반드시
`hardware_eligibility`가
`REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL`인지 확인합니다.

```bash
python paper_ready/0815/minhyuk/frozen_reference_player.py \
  --reference paper_ready/0815/flight_references/expanded/gamma_0p3_LLR_e86_seed_814323086_100hz.npz
```

위 명령은 **검증만 하고 command를 전송하지 않습니다**. 실제 hardware runner는
`stream_reference(reference, send_full_state)`를 import해서 versioned Crazyflie
sender에 연결하십시오.

## 2. Frozen reference schema

각 `.npz`에는 position만이 아니라 아래 배열이 모두 저장되어 있습니다.

```text
time_s                    (T,)    float64, 0.01 s spacing
position_ref              (T, 3)  float32
velocity_ref              (T, 3)  float32
acceleration_ref          (T, 3)  float32
raw_controls_10hz         (K, 3)  float32
applied_controls_10hz     (K, 3)  float32
executed_controls_10hz    (K, 3)  float32, applied-controls alias
hardware_eligible         ()      bool
status, method, regime, route, gamma, seed, episode
```

Hardware에서는 position을 미분해 velocity/acceleration을 다시 만들지 마십시오.
interpolation, smoothing, reference-governor 재적용도 하지 않습니다. 저장된
`position_ref`, `velocity_ref`, `acceleration_ref`를 같은 index에서 100 Hz로
전송하는 것이 contract입니다.

## 3. 승인된 40 trajectories

| View | count | status | selected narrative |
|---|---:|---|---|
| `real-paper-ready-pre2` | 8 | 8 success | `LLL×6 + RRR×2` |
| `real-paper-ready-less-expanded` | 8 | 8 success | `LLL/LLR/RLL/RRR ×2` |
| `real-paper-ready-expanded` | 8 | 8 success | approved S4 roster; corrected LLR included |
| `real-paper-ready-cfmmppi` | 8 | 6 success + 2 collision | safety 3, balanced 3, reward 2 |
| `real-paper-ready-safemppi` | 8 | 8 success | `LLL/LLR/RRL/RRR ×2` |

Expanded의 이전 negative-progress LLR seed `814321091`은 포함되지 않습니다.
대체 trajectory는 같은 LLR mode인 `814323086`이며 reverse goal step이 없습니다.

PRE2의 두 RRR은 서로 다른 실제 rollout입니다.

```text
814301373
814304759
```

모든 PRE2/R1/Expanded/SafeMPPI reference와 CFM success 6개는:

- simulated `SUCCESS`
- measured physical radius + 0.16 m hard geometry 통과
- vertical string radius 0.10 m 통과
- positive/monotone goal-progress selection 통과

입니다. 전체 40개 seed, route, status, duration, SHA-256은
`flight_references/FLIGHT_INDEX.csv`가 authoritative source입니다.

## 4. CFM–MPPI의 두 DO_NOT_FLY reference

사이트의 intended comparison을 보존하기 위해 아래 reward-dominant collision 두
개도 frozen package에는 들어 있습니다.

```text
seed 814330581, reward_dominant, COLLISION
seed 814330729, reward_dominant, COLLISION
```

두 row의 `hardware_eligibility`는 `SIMULATION_ONLY_DO_NOT_FLY`입니다.
기본 player와 `stream_reference()`는 이를 자동 거부합니다. 시각 검증만 필요하면:

```bash
python paper_ready/0815/minhyuk/frozen_reference_player.py \
  --allow-simulation-only \
  --reference <reward-dominant-collision.npz>
```

`--allow-simulation-only`는 검증-only CLI에만 적용되며 streaming interlock을
해제하지 않습니다.

## 5. 실제 설치 geometry

`scene/bowling_scene.json`과 `scene/bowling_as_built.csv`가 source of truth입니다.

| ball | x | y | z | measured radius | effective radius |
|---:|---:|---:|---:|---:|---:|
| 1 | -1.4306 | 0.7957 | 0.9514 | 0.1682 | 0.3282 |
| 2 | -1.1375 | -0.1179 | 0.9161 | 0.1788 | 0.3388 |
| 3 | -0.6215 | 0.4209 | 0.8624 | 0.1996 | 0.3596 |
| 4 | -0.8289 | -0.9135 | 0.9645 | 0.1823 | 0.3423 |
| 5 | -0.3205 | -0.4988 | 0.9285 | 0.1926 | 0.3526 |
| 6 | 0.2217 | 0.0290 | 0.9713 | 0.1653 | 0.3253 |

공 위치 또는 반경이 바뀌면 이 handoff의 safety claim은 무효입니다. 새 측정값으로
rollout과 reference export를 다시 해야 합니다.

## 6. Exact models and source

기존 0814 handoff의 checkpoints/runtime snapshot을 재사용합니다. 파일을 복제하지
않아 Git blob과 provenance가 갈라지는 것을 막았습니다.

| role | repository path | SHA-256 |
|---|---|---|
| PRE2 / CFM prior | `paper_ready/0814/checkpoints/pre2/pretrained.pt` | `76b10a69b6f26d65533d4e617ccbf6fb77a2178ac030244c5a745c11a4d0a3c0` |
| Less-expanded R1 | `paper_ready/0814/checkpoints/less_expanded/checkpoint_001.pt` | `7d5c5cc1f7e9b55ae2a803b8a3812264f38072cd74a416cd81b32b9810253f98` |
| Expanded S4 | `paper_ready/0814/checkpoints/expanded/checkpoint_004.pt` | `a37f93091bfc88340b7f2bab0d41cb889ff59a2caf00d737e58e57fbfa4cdb52` |
| Task config | `paper_ready/0815/config/task_config_resolved.json` | recorded in `SHA256SUMS` |
| Real geometry adapter | `paper_ready/0814/source/real_bowling_scene.py` | recorded in `SHA256SUMS` |
| Policy rollout | `paper_ready/0814/source/search_real_bowling_policy_rollouts.py` | recorded in `SHA256SUMS` |
| CFM–MPPI rollout | `paper_ready/0814/source/run_multisphere_cfm_mppi_bowling.py` | recorded in `SHA256SUMS` |
| SafeMPPI rollout | `paper_ready/0814/source/collect_paper_ready_safemppi_bowling.py` | recorded in `SHA256SUMS` |

`trajectories/real_selected_trajectories.pt`는 site의 full arrays를 보존합니다.
`selections/real_selection.json`은 view별 selected seed audit입니다.

## 7. Published seed를 실제 model/controller로 다시 rollout

아래는 planner를 다시 실행하는 명령입니다. Reference playback에는 필요 없습니다.
GPU/backend가 달라지면 scientific reproduction은 가능하지만 floating-point byte
identity는 보장되지 않습니다.

### PRE2

Temperature 1.2 rows:

```bash
python paper_ready/0814/source/search_real_bowling_policy_rollouts.py \
  --model pre2 --device cuda:0 --gamma 0.3 \
  --scene-json paper_ready/0815/scene/bowling_scene.json \
  --sampling-temperature 1.2 \
  --seeds 814302021 814302406 814302317 814302407 \
  --output /tmp/0815_pre2_temp12_reproduction
```

Temperature 1.4 rows, including both RRR trajectories:

```bash
python paper_ready/0814/source/search_real_bowling_policy_rollouts.py \
  --model pre2 --device cuda:0 --gamma 0.3 \
  --scene-json paper_ready/0815/scene/bowling_scene.json \
  --sampling-temperature 1.4 \
  --seeds 814304648 814304295 814301373 814304759 \
  --output /tmp/0815_pre2_temp14_reproduction
```

새 seed를 시험하려면 위 명령의 `--seeds`만 바꾸십시오. PRE2 selection bank는
temperature 1.2와 1.4를 함께 사용했으므로 비교할 때 temperature도 기록해야 합니다.

### Less-expanded / Expanded

```bash
python paper_ready/0814/source/search_real_bowling_policy_rollouts.py \
  --model less-expanded --device cuda:0 --gamma 0.3 \
  --scene-json paper_ready/0815/scene/bowling_scene.json \
  --sampling-temperature 1.0 \
  --seeds 814311152 814311187 814311146 814311039 \
          814311210 814311033 814311162 814310067 \
  --output /tmp/0815_r1_reproduction

python paper_ready/0814/source/search_real_bowling_policy_rollouts.py \
  --model expanded --device cuda:0 --gamma 0.3 \
  --scene-json paper_ready/0815/scene/bowling_scene.json \
  --sampling-temperature 1.0 \
  --seeds 814320088 814321275 814321225 814321181 \
          814321156 814321368 \
  --output /tmp/0815_s4_temp10_reproduction
```

Expanded seed `814323086`은 temperature 1.2, `814324192`는 temperature 1.4로
별도 실행합니다.

### CFM–MPPI

```bash
python paper_ready/0814/source/run_multisphere_cfm_mppi_bowling.py \
  --pretrain-dir paper_ready/0814/checkpoints/pre2 \
  --task-config paper_ready/0815/config/task_config_resolved.json \
  --as-built-scene-json paper_ready/0815/scene/bowling_scene.json \
  --output /tmp/0815_cfmmppi_reproduction --device cuda:0 \
  --gammas 0.3 --proposal-count 32 --elite-count 8 \
  --copies-per-elite 32 --mppi-sigma 0.2 --mppi-lambda 0.1 \
  --alpha-cbf 0.5 --cbf-margin-m 0.1 \
  --goal-coefficient-max 0.25 --safety-coefficient-max 1.0 \
  --regimes-json '{"safety_dominant":{"goal":0,"safety":1},"balanced":{"goal":1,"safety":1},"reward_dominant":{"goal":1,"safety":0}}' \
  --seeds 814330655 814330433 814330729 814330359 \
          814330766 814330322 814330581
```

### SafeMPPI

```bash
python paper_ready/0814/source/collect_paper_ready_safemppi_bowling.py \
  --source-root paper_ready/0814/runtime_snapshot \
  --config paper_ready/0814/config/safemppi_exact_bowling_config.json \
  --as-built-scene-json paper_ready/0815/scene/bowling_scene.json \
  --output /tmp/0815_safemppi_reproduction --device cuda:0 \
  --gammas 0.3 \
  --seeds 814341039 814341738 814341504 814341031 \
          814341870 814341342 814341765 814341082
```

## 8. Frozen references를 byte-identical하게 재생성

Planner를 호출하지 않고 저장된 full arrays에서 reference만 다시 만들려면:

```bash
REGENERATE_REFERENCES=1 bash paper_ready/0815/REPRODUCE.sh
```

40개 `.npz`가 모두 byte-identical하지 않으면 fail-close합니다.

## 9. Hardware checklist

1. `python paper_ready/0815/VERIFY.py`
2. 실제 공 center/radius가 `scene/bowling_as_built.csv`와 같은지 재측정
3. start/goal/world frame과 axis 방향 확인
4. `FLIGHT_INDEX.csv`의 `hardware_eligibility` 확인
5. validation-only player로 SHA와 first/last position 확인
6. low-altitude/low-speed dry run과 emergency stop 준비
7. operator approval 뒤에만 sender callback 연결

Simulation success와 offline clearance는 hardware safety certification이 아닙니다.
