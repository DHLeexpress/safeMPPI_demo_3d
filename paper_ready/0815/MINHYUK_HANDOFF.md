# 민혁이에게 보낼 메시지

아래 문장을 그대로 전달하면 됩니다.

---

실측 bowling 배치용 최종 flight-reference handoff를 `paper_ready/0815`에
올렸습니다. 공개 3D site의 `real-paper-ready-*` 5 views, 각 8개씩 총 40개가
들어 있고 전부 fixed `gamma=0.3`입니다.

먼저 repository root에서 실행해 주세요.

```bash
git pull origin main
python paper_ready/0815/VERIFY.py
```

비행 reference는 `paper_ready/0815/flight_references/`에 있고,
`FLIGHT_INDEX.csv`가 seed/route/status/SHA/path의 source of truth입니다. 각 `.npz`에
100 Hz `position_ref`, `velocity_ref`, `acceleration_ref`가 모두 저장돼 있으므로
position을 미분하거나 보간하지 말고 세 배열을 같은 index로 보내 주세요.

```bash
python paper_ready/0815/minhyuk/frozen_reference_player.py \
  --reference <FLIGHT_INDEX.csv에서 고른 npz>
```

이 player 자체는 validation-only라 command를 보내지 않습니다. 검증 후 기존
versioned hardware runner의 `send_full_state(position, velocity, acceleration)`에
연결하면 됩니다.

중요: `hardware_eligibility`가
`REQUIRES_OPERATOR_AND_HARDWARE_SAFETY_APPROVAL`인 row만 hardware 후보입니다.
CFM–MPPI reward-dominant 2개는 site 비교용 collision이며
`SIMULATION_ONLY_DO_NOT_FLY`로 표시했고 player도 자동 거부합니다.

실제 geometry는 `scene/bowling_as_built.csv`입니다. measured physical radius에
0.16 m를 더한 shell과, 각 공 위 수직 실 반경 0.10 m를 hard constraint로 썼습니다.
공 위치/반경이 바뀌면 그대로 날리지 말고 다시 rollout해야 합니다.

전체 설명, exact seed 재실행 명령, checkpoint/source SHA, hardware checklist는
`paper_ready/0815/README.md`에 있습니다.

---
