"""End-to-end correctness against HuggingFace, on real Llama weights. GPU required.

Compare logits and generated tokens against HuggingFace to check model math
independently of the paging tests.

