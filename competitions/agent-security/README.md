# jed-agent-security-redteam

Attack submission and working note for the Kaggle competition **AI Agent Security: Multi-Step Tool Attacks** (JED Attack, ID 134815; OpenAI / Google / IEEE).

Everything runs inside the competition's offline deterministic sandbox against fake fixtures. No real systems, credentials, or network.

## Files

| File | What it is |
|---|---|
| `attack.py` | The submission's `AttackAlgorithm`. Deterministic, self-contained, respects every SDK limit, never returns empty. This is notebook cell 2. |
| `validate_cell.py` | Local validation for a Kaggle notebook. Replays the generated candidates exactly as the gateway does, using the SDK's own scorer. Does not submit. |
| `WORKING_NOTE.md` | ~1500-word working note: method, the public-vs-private guardrail finding, results, and defender lessons. |

## What it does

`attack.py` searches for user-message chains that trip the benchmark's scored predicates (`EXFILTRATION`, `CONFUSED_DEPUTY`, `DESTRUCTIVE_WRITE`, `UNTRUSTED_TO_ACTION`). It:

- farms distinct score-cells via distinct `http.post` domains, distinct `email.send` recipients, and structural shape (the gateway keys the score-cell on tool events only, not on message wording);
- probes one representative of each family against the live model, then expands the framings that fired across many diversity values without re-probing;
- emits an **encoded-exfil** family (read `secret.txt`, then post a base64/hex/reversed form of the contents) to target the hidden private guardrail: the payload carries no secret keyword, but the `EXFILTRATION` predicate decodes it before matching.

## Results

| Setting | Candidates | Fired | normalized |
|---|---|---|---|
| Validation, public `OptimalGuardrail` | 172 | 70 | 3.9 (raw 780) |
| Validation, private-pattern stand-in | 172 | 108 | 7.1 (raw 1414) |
| Live leaderboard, public (submitted) | | | 3.9 |
| Live leaderboard, private | | | pending |

Validation numbers are a full-compliance upper bound; the private stand-in is illustrative, not the hidden guardrail.

## Reproduce

Notebook is three cells: setup, `attack.py` (written to `/kaggle/working/attack.py`), and the inference-server serve cell. The gateway writes `submission.csv` during the scored rerun, so the notebook writes none.

To sanity-check before submitting, run `validate_cell.py` inside the Kaggle notebook (it auto-locates the mounted `aicomp_sdk` and fixtures) and read the per-guardrail, per-predicate scorecard it prints.
