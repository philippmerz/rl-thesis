# Paper framing notes

Not in the main thesis until we've decided on the final scope. Parking this here for later reference.

## What generalizes from our work (the publishable parts)

1. **Reward budget inequalities H1-H4.** Math is architecture-independent. Any agent with a function approximator will hover if `w_prox > |w_hprop|/2`, etc. These are principled constraints on reward weight ratios that rule out specific degenerate policies.

2. **Bottleneck sequencing methodology.** Systematically eliminate reward design, architecture, and training dynamics as candidate bottlenecks. Applicable to other RL projects even if our specific findings don't transfer.

3. **Reproducible environment + failure mode documentation.** The env and the three reward-hacking failure modes (oscillation, hovering, suicide) provide a reusable testbed.

## What is case-study-level (don't overclaim)

- "Frame stacking helped in our setup" — specific to Rainbow-lite DQN with single-frame observations. Doesn't generalize.
- "Reward shaping is insufficient" — architecture-conditional. With a different agent, reward shaping might be exactly sufficient.
- "Constant LR prevents collapse" — consistent with literature but specific to our setup.

## Proposed contribution statement (honest)

> An empirical investigation of reward shaping in a multi-objective grid survival task. We derive four per-step design heuristics from the environment mechanics, document three reward-hacking failure modes, and show that even heuristic-satisfying reward configurations saturate below baseline for Rainbow-lite DQN with single-frame observations. We then demonstrate that temporal context (frame stacking) and training dynamics interventions (constant LR) unlock further progress, and connect the remaining gap to documented issues in the plasticity and primacy-bias literature.

This is accurate, doesn't overclaim, and still reads as a real workshop-paper contribution.

## Why this beats the earlier framing

Earlier I proposed: "Reward shaping is insufficient for multi-objective RL; the bottleneck is perceptual."

This is architecture-conditional dressed up as a general claim. With a transformer or recurrent agent, the same rewards might work fine. The user flagged this correctly: we'd be smuggling a specific finding into a general-sounding claim. The honest framing narrows the scope to what we actually tested.

## Open questions that would strengthen a real paper

- Second environment to test whether H1-H4 extend
- Theoretical bounds on when reward shaping saturates for a given architecture
- Comparison with policy gradient methods (PPO on same setup)
- Ablation of each Rainbow-lite component (do we really need Double DQN, PER, etc.?)

None of these are required for bachelor; all would be expected for a venue submission.
