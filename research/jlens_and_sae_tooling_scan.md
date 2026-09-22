# Tooling scan: J-lens and SAEs for Qwen3.5-35B-A3B (2026-09-20)

Quick scan by the main session (web search + page fetches summarised by a small model; **nothing here was read in full by me, so treat every number as to-be-verified against the source before citing**). Purpose: asri asked whether the "J lens" and SAEs could be used to see what the finetuned models are "thinking".

## J-lens (Jacobian lens)

- **Source:** Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models", transformer-circuits.pub/2026/workspace, 6 July 2026; code `github.com/anthropics/jacobian-lens` (Apache 2.0, reference implementation, unmaintained).
- **What it is:** for each layer, one d_model × d_model matrix J_l = the average Jacobian of the final-layer residual (at the same and all later positions) with respect to the layer-l residual, averaged over ~1000 generic 128-token prompts. Readout: `softmax(W_U · norm(J_l · h_l))` → a ranked list of vocabulary tokens the activation is "poised to make the model say", now or later. Differs from the logit lens by using the model's actual average input–output sensitivity instead of pretending layer l is the last layer.
- **Claims (on Claude models):** a small "J-space" (sparse non-negative combinations of per-token J-lens vectors, ≲25 active, 3–10% of activation variance) carries what the model can report and deliberately use; content outside it can be present yet not reportable. Interventions: add a token's J-lens vector, or swap one token's vector for another in lens coordinates (reported 88% success for report swaps, 54–70% for two-hop intermediates).
- **Relevant to us:** the paper's distinction "represented but not poised for report" is the same shape as our inside-vs-says lag and the coexistence finding. It also reports reading traces of finetuned-in dispositions ("reward", "bias"; "secretly", "trick") from models trained to misbehave.
- **Limits:** single-token concepts only (years are probably split into digits by the Qwen tokenizer, so "2015" vs "1944" will read badly; words like *dentist*, *gold*, *sprinter*, *hoax*, *fiction*, *false* should read fine, which favours the Dentist and Ed Sheeran claims over Vesuvius); noisy in roughly the first third of layers; by construction blind to anything not poised for verbalisation.
- **Cost to fit for our model:** one forward + ceil(d_model / dim_batch) backward passes per prompt; d_model = 2048. With dim_batch 64 that is 32 backward passes of 64 × 128 tokens per prompt. My estimate from our measured training throughput: roughly 1 minute per prompt, so ~100 prompts (reported as already usable) ≈ 1.5–2 GPU-hours, 500 prompts ≈ 8 h. Output is small (40 × 2048² floats ≈ 0.7 GB).
- **Pre-fitted lenses found:** Qwen3.6-35B-A3B (`stanleytheli/qwen3.6-35B-A3B-jlens`), Qwen3.5-122B-A10B and 397B-A17B (`dallinmj/Qwen3.5-Jacobian-Lenses`, 500 wikitext passages), Qwen3.5-9B (`bcywinski/...`); `neuronpedia/jacobian-lens` has "dozens of models", file list not checked. **None found for Qwen3.5-35B-A3B itself**, but the same hybrid MoE family has been done by others, so the architecture is not an obstacle. Third-party toolkit: `awdemos/jspace-toolkit`; nnsight has an example.
- Follow-ups seen but not read: "Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-lens" (arXiv 2608.25347); a blog post on extracting steering vectors from J-space (Sept 2026).

## SAEs

- **Qwen-Scope** (arXiv 2605.11887): official residual-stream SAEs for Qwen3 / Qwen3.5 backbones, all layers. For our model: `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100` and `...-W32K-L0_50` (TopK). **Trained on the Base model**; fit on the post-trained and LoRA-finetuned models must be checked (reconstruction error / loss recovered) before trusting features.
- Also seen: `stanleytheli/qwen3.6-35b-a3b-saes`; "Discovering Millions of Interpretable Features with Sparse Autoencoders" (arXiv 2606.26620), not read.

## Model diffing

- "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences" (arXiv 2510.13900, ICLR 2026): decoding the base-vs-finetuned activation difference (Patchscope / logit lens) on the first tokens of unrelated text recovers the finetuning topic; organisms include synthetic-document false-fact finetunes. Directly applicable to plain vs warned models.
