# Upgrade to RAGForge v2.0.3

v2.0.3 is a narrow correctness hotfix over v2.0.2.

- Focused sentence compression now preserves local entity binding for sibling labels such as `Sev-1` and `Sev-2`.
- A generic follow-up sentence inherits the immediately preceding entity for ranking, preventing a sibling value from outranking the exact entity-specific fact.
- No additional Gemini request is introduced.
- Benchmark version is `2.0.3` so saved v2.0.2 runs remain historical after the runtime evidence-selection change.
