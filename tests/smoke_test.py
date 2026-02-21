"""Quick smoke test — runs the non-LLM agent chain to verify correctness."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.ingestion import data_ingestion_agent
from src.agents.privacy_guard import privacy_mask_agent, privacy_unmask_agent
from src.agents.crime_detection import crime_detection_agent

# Load sample case
with open("data/samples/case_structuring.json", encoding="utf-8") as f:
    raw = json.load(f)

# Test ingestion
state = {"raw_data": raw}
r1 = data_ingestion_agent(state)
state.update(r1)
txn_count = len(r1["structured_data"]["transactions"])
print(f"[OK] Ingestion: {txn_count} transactions parsed")

# Test privacy mask
r2 = privacy_mask_agent(state)
state.update(r2)
mask_count = len(r2["mask_mapping"])
print(f"[OK] Privacy mask: {mask_count} PII entities masked")

# Test crime detection
r3 = crime_detection_agent(state)
state.update(r3)
crime_types = [c["type"] for c in r3["crime_types"]]
risk_types = [i["type"] for i in r3["risk_indicators"]]
print(f"[OK] Crime detection: types={crime_types}")
print(f"     Risk indicators: {risk_types}")

# Test typology agents
from src.agents.typology.transaction_fraud import transaction_fraud_agent
from src.agents.typology.country_risk import country_risk_agent
from src.agents.typology.text_content import text_content_agent
from src.agents.typology.payment_velocity import payment_velocity_agent
from src.agents.typology.account_health import account_health_agent

state["typology_results"] = {}
for name, func in [
    ("transaction_fraud", transaction_fraud_agent),
    ("country_risk", country_risk_agent),
    ("text_content", text_content_agent),
    ("payment_velocity", payment_velocity_agent),
    ("account_health", account_health_agent),
]:
    r = func(state)
    state.update(r)
    findings = state["typology_results"][name]["findings"]
    score = state["typology_results"][name]["risk_score"]
    print(f"[OK] Typology[{name}]: {len(findings)} findings, score={score:.2f}")

# Test graph build
from src.graph.sar_graph import build_sar_graph
app = build_sar_graph(interrupt_before=[])
print(f"[OK] SAR graph built successfully")

print("\n=== ALL SMOKE TESTS PASSED ===")
