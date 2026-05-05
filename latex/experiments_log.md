# Experiments Log

Chronological record of all reward shaping and architecture experiments. Each entry states motivation, key changes, result, and takeaway.

## Final ablation results

The thesis reports the reward x observation ablation below. All training is
2M steps, 3 seeds per cell (seeds 42, 43, 44), pure RL with no replay-buffer
demonstrations. Final benchmarks are 100 episodes on seeds 1000-1099. The
heuristic baseline reaches 768 +/- 19 ticks survival.

```
Cell                       n  surv mean  food  death rate  diff vs H  sig losses
Baseline (sf)              3  651        0.70  98%         -118       3 of 3
Baseline (fs)              3  548        5.01  96%         -221       3 of 3
Absolute proximity (sf)    3  640        0.38  99%         -129       3 of 3
Absolute proximity (fs)    3  591        4.76  97%         -178       3 of 3
Engineered V5 (sf)         3  670        0.09 100%          -98       3 of 3
Engineered V5 (fs)         3  744        2.15  94%          -24       1 of 3
Weak proximity (sf, 1 seed) 1 647        0.66  99%         -122       1 of 1
```

Key findings:
- No DQN cell significantly outperforms the heuristic.
- The closest cell is Engineered V5 with frame stacking, which reaches
  statistical parity in 2 of 3 seeds and significantly loses in 1 of 3.
- Frame stacking has a reward-dependent effect: it improves Engineered V5
  by +74 ticks, hurts Absolute proximity by -49, and hurts Baseline by -103.

The historical log below records earlier (often demo-tainted) experiments;
those numbers should not be cited in the final thesis. The final thesis
relies only on the table above and the cells stored under `runs/`.

## Reference baselines

**Heuristic baseline** (scripted agent)
- 768 mean survival, 1.19 food, 95% death rate, 5% time-limit (1000 tick)
- Camps in nearest shelter, forages when hunger < 0.5, flees enemies within distance 5