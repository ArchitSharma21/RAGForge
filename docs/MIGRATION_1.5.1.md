# Migration to RAGForge v1.5.1

v1.5.1 is a stabilization patch over v1.5. It does not change dependency pins or the bundled NIST
PDF.

## Fixes

- Saved Quick/Standard/Deep reports now render as proper formatted JSON after switching evaluation depth.
- Saved-report persistence/API output is normalized to plain JSON-compatible values.
- Every evaluation table can be copied/exported as CSV, TSV or Markdown.
- Grouped citations such as `[D1, D2]` are recognized by evaluation and normalized for display.
- Zero-call citation repair skips broad preamble/list-introduction lines ending in `:`.
- Redundant trailing citation groups are collapsed.

## Cache compatibility

The demo benchmark version is bumped from `1.5` to `1.5.1` because citation evaluation semantics
changed. Existing v1.5 saved reports should therefore be treated as historical rather than reused as
current v1.5.1 benchmark results.

## Deployment

Apply the patch over v1.5 and commit normally. The patch does not include
`demo_documents/NIST_AI_RMF_1.0.pdf`, so the existing Hugging Face Xet/LFS setup is unchanged.
