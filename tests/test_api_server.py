"""OpenAI-compatible server tests, driven over real HTTP against the CPU fixture model.

The server is the thing the benchmark client actually measures, so a bug here is a bug
in every number the project reports. The two that matter most: streaming must emit one
chunk per token (or TTFT and ITL are fiction), and requests must be served
concurrently by the background engine loop rather than serialised by the handlers.
"""

