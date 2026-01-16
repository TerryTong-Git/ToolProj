"""
exps_control: Controlled experiments for CoT prefilling comparison.

This experiment tests whether translated code reasoning performs similarly
to original NL reasoning when prefilled as Chain-of-Thought.

Experiment design:
1. Load existing results with generated code (sim_code) and NL reasoning (nl_reasoning)
2. Translate code to NL using GPT 5.2 with the consistency-optimized prompt
3. Create two conditions:
   - Condition A: Original problem + original NL reasoning (prefilled) → answer
   - Condition B: Original problem + translated code-to-NL (prefilled) → answer
4. Compare accuracy between conditions
"""
