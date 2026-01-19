from src.acti_router.core.schemas import RouteResult

def test_route_result_schema_validation():
    sample = {
        "source": "n8n",
        "workflow_id": "abc123",
        "workflow_name": "Example",
        "confidence": 0.75,
        "reasoning": "Matches memory retrieval and context storage requirements.",
        "matched_signals": ["memory retrieval", "context storage"]
    }
    rr = RouteResult(**sample)
    assert rr.workflow_id == "abc123"
    assert 0 <= rr.confidence <= 1
    assert isinstance(rr.matched_signals, list)

def test_route_result_rejects_out_of_range_confidence():
    bad = {
        "source": "n8n",
        "workflow_id": "abc123",
        "workflow_name": "Example",
        "confidence": 1.5,
        "reasoning": "nope",
        "matched_signals": []
    }
    try:
        RouteResult(**bad)
        assert False, "Expected validation error"
    except Exception:
        assert True
