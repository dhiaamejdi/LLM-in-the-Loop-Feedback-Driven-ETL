# Ablation summary (pooled across seeds)

- **kb / Circular (same-family LLM)** (3 seeds): TP=43 FN=8 FP=45 TN=24 | recall 43/51 = 0.843 CI [0.714, 0.930] | precision 0.489 | unscored rows 15
- **kb / Reflexion (same-family + self-critique)** (3 seeds): TP=43 FN=8 FP=39 TN=30 | recall 43/51 = 0.843 CI [0.714, 0.930] | precision 0.524 | unscored rows 15
- **kb / Deterministic rules only** (3 seeds): TP=17 FN=34 FP=0 TN=69 | recall 17/51 = 0.333 CI [0.208, 0.479] | precision 1.000
- **kb / Cross-architecture LLM only** (3 seeds): TP=11 FN=40 FP=0 TN=69 | recall 11/51 = 0.216 CI [0.113, 0.353] | precision 1.000
- **kb / Hybrid (rules + cross-architecture)** (3 seeds): TP=26 FN=25 FP=0 TN=69 | recall 26/51 = 0.510 CI [0.366, 0.652] | precision 1.000
- **kb / Hybrid (rules + same-family)** (3 seeds): TP=42 FN=9 FP=47 TN=22 | recall 42/51 = 0.824 CI [0.691, 0.916] | precision 0.472 | unscored rows 15
- **kb_water / Deterministic rules only** (10 seeds): TP=796 FN=0 FP=0 TN=2204 | recall 796/796 = 1.000 CI [0.995, 1.000] | precision 1.000
- **kb_water_oracle / Deterministic rules only** (10 seeds): TP=15 FN=0 FP=0 TN=2985 | recall 15/15 = 1.000 CI [0.782, 1.000] | precision 1.000
- **kbfull / Deterministic rules only** (10 seeds): TP=9102 FN=0 FP=0 TN=898 | recall 9102/9102 = 1.000 CI [1.000, 1.000] | precision 1.000
- **uci / Circular (same-family LLM)** (3 seeds): TP=49 FN=4 FP=27 TN=10 | recall 49/53 = 0.925 CI [0.818, 0.979] | precision 0.645
- **uci / Reflexion (same-family + self-critique)** (3 seeds): TP=46 FN=7 FP=25 TN=12 | recall 46/53 = 0.868 CI [0.747, 0.945] | precision 0.648
- **uci / Deterministic rules only** (3 seeds): TP=15 FN=38 FP=0 TN=37 | recall 15/53 = 0.283 CI [0.168, 0.423] | precision 1.000
- **uci / Cross-architecture LLM only** (3 seeds): TP=15 FN=38 FP=1 TN=36 | recall 15/53 = 0.283 CI [0.168, 0.423] | precision 0.938
- **uci / Hybrid (rules + cross-architecture)** (3 seeds): TP=24 FN=29 FP=1 TN=36 | recall 24/53 = 0.453 CI [0.316, 0.596] | precision 0.960
- **uci / Hybrid (rules + same-family)** (3 seeds): TP=51 FN=2 FP=21 TN=16 | recall 51/53 = 0.962 CI [0.870, 0.995] | precision 0.708
