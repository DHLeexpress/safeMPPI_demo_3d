# Source and artifact provenance

Pinned software snapshot:
[`dabb5011dfc674864e1de275a1e1c2adab58f4af`](https://github.com/DHLeexpress/safeMPPI_demo_3d/tree/dabb5011dfc674864e1de275a1e1c2adab58f4af).

| role | repository path | file SHA-256 | last-touch commit |
|---|---|---|---|
| cylinder-ID distribution | `configs/lab_clutter_cylinders_path_midpoint_uniform_v2.json` | `9336bc73ff6d18e2cfafe8f96db455ae9c7614a8c981906d196da7c385591429` | `c46ae2fff7f77b920d38c2e249c43267f68504ca` |
| five-cylinder generator | `configs/lab_clutter_cylinders_lab_five_v2.json` | `774e1c9bb2e2636b75e3ef46253dd612bdb903070026b05a4b888adb51e6d989` | pinned by snapshot |
| SafeMPPI controller | `safe_mppi/controller.py` | `dfc91a26ccac2818c902215bf4d9a06e405d5878e5c6af0be2f75c4f68106dad` | `1d9fb1bcac282e47144239c14ec239e261cb0f89` |
| randomized scene implementation | `safe_mppi/path_focused_clutter.py` | `e4011c9980824a7d89baefa914822505d2bc7cacfdf224ed4589b34684fce6dc` | pinned by snapshot |
| rollout collection | `safe_mppi/path_focused_collection.py` | `eea83bfaf1630a6428f47d903f13ad86afa89e5c813f3b8d30ac62147ce90a71` | pinned by snapshot |
| audit/success-quota runner | `scripts/collect_path_focused_success_quota.py` | `54e835d873e49186586add0ee8b714cef8c8d1e2bda604fc3a0455d4bb031561` | pinned by snapshot |
| pretrained checkpoint | `flow_deployment/minhyuk_stage1_handoff/checkpoints/hp100_t128_d3.pt` | `cc87b65f27506254509b7f4cbbe4734aacfc9e50640a3756cfb0b1ed456e28ff` | `ef271a8d3e4df4eb01194265162e77af8249aa31` |
| model contract | `flow_deployment/minhyuk_stage1_handoff/model_contract.json` | `0096894e37c9496231110159ec38267e9d555f0f8e03a80edea202ebfc8ceef8` | pinned by snapshot |
| pretrained loader | `flow_deployment/lab_pretrained.py` | `0218f8c3f5691c7875370974783aa862b4140594e8c75ab4435655448a353b79` | pinned by snapshot |
| HP100 policy/encoder | `safe_mppi/lab_visual_flow.py` | `f93139d8f3f9f2ed4554b8d9c5dafbd948ada784ad5e53c52af161ce82fa3c1b` | pinned by snapshot |
| closed-loop flow task | `safe_mppi/lab_reference_flow_task.py` | `0b3058908cb6aefd15f002e858c3b213c51a8c27bfbb5e62ded5ebdcc9d98b92` | pinned by snapshot |
| deployment software smoke | `scripts/run_lab_flow_deployment.py` | `cd5c9bc4a718aa96ccf93f8d626e8f56e5c2af12d26482eb3dbf7a6f82e1ec0e` | `58ed7c68580abf59c9dccf1b9da3f1c34938d6d6` |
| frozen-reference exporter | `scripts/export_lab_flow_frozen_references.py` | `5928ea72ef3ec73b47836e22f0b45e1d9c5bc181ce654024e65eed6641b05acc` | pinned by snapshot |
| offline deployment harness | `deploy_sim/run_offline.py` | `0bd338d040aecb4a83d6714f178175a1c9f27f5bfe8637da3bb405f35d4a42e3` | pinned by snapshot |
| governor/smoothing/geofence | `deploy_sim/harness.py` | `da7f21c06c2ddd1f2be123c8375cd04cc2c2c1b0b82abd499c90b72778ffccf8` | pinned by snapshot |
| reference-following vehicle | `deploy_sim/vehicle.py` | `030f7836862297156aae92765c0f676a06f44ed9f238da795614c9edd2b3d077` | pinned by snapshot |

Pretrained policy contract: 8,000 accepted trajectories from 3,262 randomized
4--8-cylinder scenes; selected epoch 52; `1 x 32 x 32 x 100` clipped-
`H_P` visual input; 128-dimensional visual token; depth-3 flow trunk;
`H=10` by 3-D acceleration output; no GRU. The source archive SHA-256 is
`c9e593896678f1a083af27cd56b50c509bcc5b4670ed1393cd1e3d5679d58602`.

`deploy_sim/` is an offline software harness, not the hardware logger. Every
paper flight must record the actual Crazyflie/Vicon runner path and commit;
the previous Drive campaign did not.
