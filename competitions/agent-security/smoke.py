from pathlib import Path
from aicomp_sdk.agents import build_agent
from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.attacks.baselines.attacker_goexplore import AttackAlgorithm
from aicomp_sdk.attacks.contracts import AttackRunConfig

fixtures = Path("data/aicomp_sdk/fixtures")
env = SandboxEnv(seed=123, fixtures_dir=fixtures,
                 agent=build_agent("deterministic"), guardrail=Guardrail())
atk = AttackAlgorithm({"max_turns": 12, "branch_batch": 16})
findings = atk.run(env, AttackRunConfig(time_budget_s=60))

print("\n===== RESULT =====")
print("findings:", len(findings))
for i, d in enumerate(atk.last_run_details[:8], 1):
    preds = [p["predicate"] for p in d["predicates"]]
    print(f"[{i}] {preds}  chain={d['user_messages']}")
