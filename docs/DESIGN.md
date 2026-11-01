# Design

Written for someone who knows transformers but not serving. It covers the three things
that make an inference engine an *engine* rather than a `generate()` loop: how KV
memory is managed, how work is scheduled, and how the attention kernel copes with the
first two.

---

## 0. The shape of the system

```
