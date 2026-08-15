# Expanded S4 · 8-mode completion backup references

이 폴더는 기존 `paper_ready/0815/flight_references/` 40개를 수정하거나
대체하지 않습니다. `real-paper-ready-expanded`의 중복 `LLL` 1개와 `RRR`
1개 대신 사용할 **추가 비행 reference 2개**입니다.

## 민혁이에게 전달할 실행 순서

1. 먼저 `LRR` reference를 검증하고 비행합니다.
2. 다음으로 `RLR` reference를 검증하고 비행합니다.
3. 두 파일 모두 저장된 100 Hz position/velocity/acceleration을 그대로
   전송합니다. position에서 velocity/acceleration을 다시 미분하지 않습니다.
4. interpolation, smoothing, Reference Governor 재적용을 하지 않습니다.
5. 기존 0815 handoff와 마찬가지로 operator/hardware safety approval 뒤에만
   비행합니다.

| 순서 | mode | seed | temperature | reference |
|---:|---|---:|---:|---|
| 1 | LRR | `815145100` | `1.40` | `gamma_0p3_LRR_seed_815145100_100hz.npz` |
| 2 | RLR | `815135084` | `1.35` | `gamma_0p3_RLR_seed_815135084_100hz.npz` |

검증 예시:

```bash
python paper_ready/0815/minhyuk/frozen_reference_player.py \
  --reference paper_ready/0815/backup_references/expanded_8mode_completion_v1/gamma_0p3_LRR_seed_815145100_100hz.npz

python paper_ready/0815/minhyuk/frozen_reference_player.py \
  --reference paper_ready/0815/backup_references/expanded_8mode_completion_v1/gamma_0p3_RLR_seed_815135084_100hz.npz
```

두 validation 명령은 `commands_sent: 0`인 dry validation입니다. 실제 송신은
기존 0815 hardware runner에 각 reference의 `position_ref`, `velocity_ref`,
`acceleration_ref`를 100 Hz로 연결해야 합니다.

## 선택 gate

두 trajectory는 모두 다음을 통과했습니다.

- Expanded S4 checkpoint, fixed gamma `0.3`, faithful raw NFE16·M1
- 실측 bowling sphere + physical radius별 `0.16 m` effective margin
- physical sphere 상단부터 반경 `0.10 m` vertical-string no-go
- hard z-band occupancy `>= 0.90`
- terminal goal-progress gate; 두 trajectory 모두 reverse step `0`
- simulation status `SUCCESS`

세부 수치와 모든 artifact SHA256은 `manifest.json`, `FLIGHT_INDEX.csv`,
`SHA256SUMS`에 고정되어 있습니다. 원본 두 trajectory는
`selected_trajectories.pt`에 함께 보존했습니다.

## 최종 site roster

사이트의 Expanded view는 아래 mode를 정확히 한 개씩 사용합니다.

```text
LLL  LLR  LRL  LRR  RLL  RLR  RRL  RRR
```

이 backup에서 `LRR`과 `RLR`을 공급하며, 기존 site 중복 seed
`814321156` (LLL)과 `814321368` (RRR)은 site 표시에서만 제외합니다.
기존 GitHub code와 기존 40-reference handoff는 변경하지 않습니다.
