# Migration: v1.1 -> v1.2

Replace the v1.1 source/docs with the v1.2 patch and push a normal new commit. No database migration is required because workspaces are ephemeral.

## Important behavioral changes

- `gr.State` session ID -> `gr.BrowserState` with compatibility fallback.
- `Workspace.ingest()` accepts an optional `progress_callback`. Existing callers remain valid.
- query traces now include `workspace` metadata.
- retrieval trace separates unique `selected_sources` from `retrieved_chunks`.
- the graph has an explicit `abstain` terminal node.
- UI demo mode can auto-index the bundled corpus when the workspace is empty.
- source snippets are stored longer for diagnostics but rendered as clean word-boundary previews.

## Deployment

The v1.1 dependency pins and Hugging Face filesystem permission fix are retained. The NIST PDF remains tracked through Git Xet/LFS; do not remove `.gitattributes`.

After deploy, test:

1. fresh page + demo enabled -> ask `What is the collection about?` without manually indexing; demo should initialize and answer from documents.
2. refresh page -> corpus should remain available while the same Space process is alive.
3. reset session + disable demo -> a document-only query should return the explicit no-corpus abstention without retrieval/verification loops.
4. manual indexing -> button should show `Building corpus...` and progress stages.
5. source panel -> long snippets should end with `...` rather than a cut-off word.
