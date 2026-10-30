# DeepSeek V4.1 unified serving additions

The serving fork starts at upstream
`f970d0b3a97c59481436a542ed8c0d73a731ace5`. Prefill and Decode use the same
package, without restoring the old Prefill vendor or Decode 0.1.12 package.

Selected Decode FP8 projections set `stream_k_max_slices=3` in their tuning
configuration, with `use_tma_c=False`. The scheduler accounts for a partial
first slice when enforcing the limit. This retains the ordered BF16 reduction
path in `GmemWriter::write_legacy`; other callers keep the upstream ten-slice
limit. The limit is part of the JIT configuration and kernel cache identity.

The original optimization is vLLM
`9f2271588e57434ea93d3e9ceb7e13b644f79250`, which used a Python partition check
against Humming 0.1.12. Current callers must use the native limit instead of
assuming that the old scheduler formula still describes upstream scheduling.

The upstream WGMMA split issue/wait and late scale prefetch code is retained.
Prefill W13 tuning should enable those native fields rather than replacing
the consumer pipeline with the old vendored implementation.

Original performance claims require new numerical and serving validation on
the unified package.
