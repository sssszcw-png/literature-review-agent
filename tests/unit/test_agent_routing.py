"""Tests for LangGraph routing logic."""

from src.agent.routing import route_after_detect_gaps, route_after_generate_outline


class TestRouteAfterDetectGaps:
    def test_round1_broad_goes_to_plan_queries(self, agent_state_broad):
        result = route_after_detect_gaps(agent_state_broad)
        assert result == "plan_queries"

    def test_deep_dive_no_saturation_goes_to_evaluate(self, agent_state_broad):
        state = {**agent_state_broad, "phase": "deep_dive", "current_round": 2}
        state["gaps"] = [
            {"description": "gap1", "severity": "critical", "saturation": 0.0}
        ]
        result = route_after_detect_gaps(state)
        assert result == "evaluate_saturation"

    def test_all_saturated_goes_to_outline(self, agent_state_broad):
        state = {**agent_state_broad, "phase": "deep_dive", "current_round": 2}
        state["gaps"] = [
            {"description": "gap1", "severity": "critical", "saturation": 0.85},
            {"description": "gap2", "severity": "important", "saturation": 0.9},
        ]
        result = route_after_detect_gaps(state)
        assert result == "generate_outline"

    def test_max_rounds_exceeded_goes_to_outline(self, agent_state_broad):
        state = {
            **agent_state_broad,
            "phase": "deep_dive",
            "current_round": 5,
            "max_rounds": 5,
        }
        state["gaps"] = [
            {"description": "gap1", "severity": "critical", "saturation": 0.3}
        ]
        result = route_after_detect_gaps(state)
        assert result == "generate_outline"

    def test_no_improvement_goes_to_outline(self, agent_state_broad):
        state = {
            **agent_state_broad,
            "phase": "deep_dive",
            "current_round": 3,
            "consecutive_no_improvement": 2,
        }
        state["gaps"] = [
            {"description": "gap1", "severity": "critical", "saturation": 0.3}
        ]
        result = route_after_detect_gaps(state)
        assert result == "generate_outline"

    def test_critical_unsaturated_continues(self, agent_state_broad):
        state = {
            **agent_state_broad,
            "phase": "deep_dive",
            "current_round": 2,
            "consecutive_no_improvement": 0,
        }
        state["gaps"] = [
            {"description": "gap1", "severity": "critical", "saturation": 0.3}
        ]
        result = route_after_detect_gaps(state)
        assert result == "plan_queries"

    def test_only_non_critical_unsaturated(self, agent_state_broad):
        state = {
            **agent_state_broad,
            "phase": "deep_dive",
            "current_round": 2,
        }
        state["gaps"] = [
            {"description": "gap1", "severity": "important", "saturation": 0.3},
            {"description": "gap2", "severity": "nice_to_have", "saturation": 0.1},
        ]
        result = route_after_detect_gaps(state)
        # No critical gaps → go to outline, even if important/nice_to_have are unsaturated
        assert result == "generate_outline"


class TestRouteAfterGenerateOutline:
    def test_approve(self, agent_state_broad):
        state = {**agent_state_broad, "user_action": "approve"}
        assert route_after_generate_outline(state) == "write_report"

    def test_edit(self, agent_state_broad):
        state = {**agent_state_broad, "user_action": "edit"}
        assert route_after_generate_outline(state) == "generate_outline"

    def test_abort(self, agent_state_broad):
        state = {**agent_state_broad, "user_action": "abort"}
        assert route_after_generate_outline(state) == "__end__"

    def test_default_action(self, agent_state_broad):
        state = {**agent_state_broad}
        state.pop("user_action", None)
        assert route_after_generate_outline(state) == "write_report"
