from src.acti_router.core.catalog import build_profile

def test_profile_contains_core_fields():
    obj = {"nodes": [], "edges": []}
    p = build_profile(obj, "bland", "wf1", "MyWF")
    assert "WORKFLOW_ID: wf1" in p
    assert "NAME: MyWF" in p
    assert "SOURCE: bland" in p
