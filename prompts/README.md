# Writer prompts

The task files that writers received, copied verbatim from Claude Opus 5.5 runs. GPT-6.1 Sol and GPT-6 Astra received the same task files.

| File | Task |
|---|---|
| `free-arc_gen1.md` | ARC, free design, generation 1 |
| `free-arc_gen2.md` | ARC, free design, generation 2 (with the parent model and its validation accuracy) |
| `mem_assoc2-arc_gen1.md` | ARC, our associative-memory-based model, generation 1 |
| `transformer-arc.md` | ARC, the given two-layer Transformer |
| `lm-init_qwen3_1_7b.md` | ARC, a released architecture from random initialization (Qwen3-1.7B) |
| `lm-pre_qwen3_1_7b.md` | ARC, a released architecture from its pretrained checkpoint (Qwen3-1.7B) |

Each run starts with the single instruction

> Read /srv/work/TASK.md and complete the task described there.

When the writer replies that it is done, it is asked once more, until a stopping condition holds (Appendix A of the paper):

> Review your model against the task and improve /srv/work/model under the same rules if you can, then reply when done.

The noun and the path follow the deliverable (`model`, `memory`, `network`).
