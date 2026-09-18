"""nnprobe: probing toolkit for the Negation Neglect representational study.

Modules
- model:   load Qwen3.5-35B-A3B, residual-stream hooks on decoder layers
- acts:    last-token activations for statement lists; span activations inside long documents
- ttpd:    port of Bürger et al. (2024) truth×polarity probe (Truth is Universal, probes.py)
- data:    TTPD topic datasets, claim statement set, SDF document cells, claim-sentence localisation
- logprob: association logprob readout (teacher-forced answer logprob under prefixes)
"""
