from src.acti_router.core.catalog import detect_source

def test_detect_source_bland():
    obj = {"nodes": [], "edges": []}
    assert detect_source(obj) == "bland"

def test_detect_source_n8n():
    obj = {"id": "123", "nodes": [], "connections": {}}
    assert detect_source(obj) == "n8n"

def test_detect_source_unknown():
    obj = {"foo": "bar"}
    assert detect_source(obj) == "unknown"
