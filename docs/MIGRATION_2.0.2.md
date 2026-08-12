# Upgrade to RAGForge v2.0.2

v2.0.2 is a final correctness hotfix over v2.0.1.

## What changed

- Numeric QA matching is contradiction-aware: a wrong primary value cannot pass because the correct value appears later in a disclaimer or correction.
- The generation instruction explicitly binds nearby numeric values to their correct entity or condition and avoids contradictory primary answers.
- Benchmark version is `2.0.2`, so saved `2.0.1` reports remain historical rather than being reused under changed scoring semantics.

No runtime dependency or retrieval-policy change is introduced.
