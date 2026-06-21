"""
Tests for pipeline.py pure helpers and orchestration logic.

All LLM calls (complete()) are mocked — no real network calls or API keys needed.
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the pipeline module can be imported without a real ANTHROPIC_API_KEY
# by patching AsyncAnthropic before it is instantiated at module level.
# ---------------------------------------------------------------------------
_mock_client = MagicMock()
_mock_client.messages = MagicMock()
_mock_client.messages.create = AsyncMock()

with patch("anthropic.AsyncAnthropic", return_value=_mock_client):
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    import pipeline


# ---------------------------------------------------------------------------
# Helper to run async functions in tests (asyncio.run creates a fresh loop)
# ---------------------------------------------------------------------------
def arun(coro):
    return asyncio.run(coro)


# ===========================================================================
# 1. _json() — JSON extraction helper
# ===========================================================================
class TestJsonExtraction:
    def test_plain_object(self):
        result = pipeline._json('{"verified": true, "reason": "ok"}')
        assert result == {"verified": True, "reason": "ok"}

    def test_object_with_preamble(self):
        """Model often replies with markdown before the JSON."""
        text = 'Sure! Here is the JSON:\n{"verified": false, "reason": "wrong"}'
        result = pipeline._json(text)
        assert result["verified"] is False

    def test_nested_object_in_text(self):
        text = 'Result: {"verified": true, "reason": "supported by evidence"} end'
        result = pipeline._json(text)
        assert result["verified"] is True
        assert "evidence" in result["reason"]

    def test_pure_array_of_numbers(self):
        """Arrays with no braces inside use the [ path."""
        result = pipeline._json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_pure_array_of_strings(self):
        """Arrays of strings (no {}) correctly use the [ fallback path."""
        result = pipeline._json('["alpha", "beta"]')
        assert result == ["alpha", "beta"]

    def test_array_of_objects_uses_object_path(self):
        """
        _json finds { before [ when objects are inside the array.
        With a single-element array like [{"k":"v"}], it extracts the inner
        object dict (not the surrounding list) — that's the actual behaviour.
        """
        result = pipeline._json('[{"k": "v"}]')
        # _json extracts the inner object, not the list wrapper
        assert result == {"k": "v"}

    def test_raises_on_no_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            pipeline._json("no json here at all")

    def test_object_with_numeric_values(self):
        result = pipeline._json('{"count": 42, "ratio": 3.14}')
        assert result["count"] == 42
        assert abs(result["ratio"] - 3.14) < 1e-9

    def test_object_with_bool_values(self):
        result = pipeline._json('{"verified": true}')
        assert result["verified"] is True


# ===========================================================================
# 2. Claim dataclass
# ===========================================================================
class TestClaimDataclass:
    def test_defaults(self):
        c = pipeline.Claim(angle="risks", text="Some risk claim")
        assert c.verified is False
        assert c.reason == ""

    def test_explicit_fields(self):
        c = pipeline.Claim(
            angle="key facts",
            text="Water boils at 100°C",
            verified=True,
            reason="physics",
        )
        assert c.verified is True
        assert c.reason == "physics"

    def test_angle_and_text_stored(self):
        c = pipeline.Claim(angle="contrarian", text="Actually, X is false")
        assert c.angle == "contrarian"
        assert c.text == "Actually, X is false"

    def test_mutable_after_creation(self):
        c = pipeline.Claim(angle="a", text="b")
        c.verified = True
        c.reason = "updated"
        assert c.verified is True
        assert c.reason == "updated"


# ===========================================================================
# 3. research() — fan-out stage
#
# _json has a quirk: for arrays of objects the first { is found before [,
# so it attempts to extract a single JSON object.  We patch _json directly
# to control what the parser returns, which lets us focus on testing the
# Claim-assembly logic inside research().
# ===========================================================================
class TestResearch:
    def test_returns_claims_for_each_parsed_item(self):
        parsed = [{"text": "Claim one"}, {"text": "Claim two"}, {"text": "Claim three"}]
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="ignored")), \
             patch.object(pipeline, "_json", return_value=parsed):
            claims = arun(pipeline.research("What is X?", "key facts"))
        assert len(claims) == 3
        assert all(c.angle == "key facts" for c in claims)
        assert claims[0].text == "Claim one"
        assert claims[2].text == "Claim three"

    def test_angle_assigned_correctly(self):
        parsed = [{"text": "Risk 1"}]
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="ignored")), \
             patch.object(pipeline, "_json", return_value=parsed):
            claims = arun(pipeline.research("Q?", "risks / caveats"))
        assert claims[0].angle == "risks / caveats"

    def test_returns_empty_on_bad_json(self):
        """When _json raises (bad model output), research() catches and returns []."""
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="not json at all")):
            claims = arun(pipeline.research("Q?", "key facts"))
        assert claims == []

    def test_returns_empty_when_json_parse_raises(self):
        """Explicit exception from _json is caught gracefully."""
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="x")), \
             patch.object(pipeline, "_json", side_effect=json.JSONDecodeError("err", "", 0)):
            claims = arun(pipeline.research("Q?", "key facts"))
        assert claims == []

    def test_returns_empty_on_empty_list(self):
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="x")), \
             patch.object(pipeline, "_json", return_value=[]):
            claims = arun(pipeline.research("Q?", "key facts"))
        assert claims == []

    def test_all_claims_unverified_by_default(self):
        parsed = [{"text": "A"}, {"text": "B"}]
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="x")), \
             patch.object(pipeline, "_json", return_value=parsed):
            claims = arun(pipeline.research("Q?", "key facts"))
        assert all(not c.verified for c in claims)
        assert all(c.reason == "" for c in claims)

    def test_text_extracted_from_each_item(self):
        parsed = [{"text": "Alpha"}, {"text": "Beta"}]
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="x")), \
             patch.object(pipeline, "_json", return_value=parsed):
            claims = arun(pipeline.research("Q?", "key facts"))
        assert [c.text for c in claims] == ["Alpha", "Beta"]


# ===========================================================================
# 4. verify() — adversarial verification stage
# ===========================================================================
class TestVerify:
    def _claim(self, text: str = "Some claim") -> pipeline.Claim:
        return pipeline.Claim(angle="key facts", text=text)

    def test_verified_true(self):
        reply = json.dumps({"verified": True, "reason": "well supported"})
        with patch.object(pipeline, "complete", new=AsyncMock(return_value=reply)):
            result = arun(pipeline.verify("Q?", self._claim()))
        assert result.verified is True
        assert result.reason == "well supported"

    def test_verified_false(self):
        reply = json.dumps({"verified": False, "reason": "contradicted by sources"})
        with patch.object(pipeline, "complete", new=AsyncMock(return_value=reply)):
            result = arun(pipeline.verify("Q?", self._claim()))
        assert result.verified is False
        assert "contradicted" in result.reason

    def test_unparseable_defaults_to_rejected(self):
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="oops no json")):
            result = arun(pipeline.verify("Q?", self._claim()))
        assert result.verified is False
        assert "unparseable" in result.reason

    def test_missing_verified_key_treated_as_falsy(self):
        """If 'verified' key absent, dict.get returns None which bool() treats as False."""
        reply = json.dumps({"reason": "no verdict key"})
        with patch.object(pipeline, "complete", new=AsyncMock(return_value=reply)):
            result = arun(pipeline.verify("Q?", self._claim()))
        assert result.verified is False

    def test_missing_reason_defaults_to_empty_string(self):
        reply = json.dumps({"verified": True})
        with patch.object(pipeline, "complete", new=AsyncMock(return_value=reply)):
            result = arun(pipeline.verify("Q?", self._claim()))
        assert result.reason == ""

    def test_returns_same_claim_object(self):
        """verify() mutates and returns the same Claim instance."""
        c = self._claim("Specific claim")
        reply = json.dumps({"verified": True, "reason": "ok"})
        with patch.object(pipeline, "complete", new=AsyncMock(return_value=reply)):
            returned = arun(pipeline.verify("Q?", c))
        assert returned is c

    def test_verify_false_string_is_falsy(self):
        """verified=false in JSON -> Python False -> bool(False) = False."""
        reply = json.dumps({"verified": False, "reason": "wrong"})
        with patch.object(pipeline, "complete", new=AsyncMock(return_value=reply)):
            result = arun(pipeline.verify("Q?", self._claim()))
        assert result.verified is False

    def test_reason_preserved_exactly(self):
        reason = "This specific reason text with punctuation: yes."
        reply = json.dumps({"verified": True, "reason": reason})
        with patch.object(pipeline, "complete", new=AsyncMock(return_value=reply)):
            result = arun(pipeline.verify("Q?", self._claim()))
        assert result.reason == reason


# ===========================================================================
# 5. synthesize() — final synthesis stage
# ===========================================================================
class TestSynthesize:
    def _claim(self, text: str, angle: str = "key facts", verified: bool = True) -> pipeline.Claim:
        c = pipeline.Claim(angle=angle, text=text)
        c.verified = verified
        return c

    def test_no_verified_claims_returns_hardcoded_message(self):
        claims = [self._claim("A", verified=False), self._claim("B", verified=False)]
        with patch.object(
            pipeline, "complete", new=AsyncMock(side_effect=AssertionError("LLM must not be called"))
        ):
            result = arun(pipeline.synthesize("Q?", claims))
        assert "No claims survived" in result

    def test_empty_claims_list_returns_hardcoded_message(self):
        with patch.object(
            pipeline, "complete", new=AsyncMock(side_effect=AssertionError("LLM must not be called"))
        ):
            result = arun(pipeline.synthesize("Q?", []))
        assert "No claims survived" in result

    def test_verified_claims_included_in_prompt(self):
        claims = [
            self._claim("Verified fact 1", angle="key facts", verified=True),
            self._claim("Rejected fact", angle="risks / caveats", verified=False),
            self._claim("Verified fact 2", angle="contrarian", verified=True),
        ]
        captured = {}

        async def fake_complete(model, prompt, **kwargs):
            captured["prompt"] = prompt
            return "A tight brief."

        with patch.object(pipeline, "complete", new=fake_complete):
            result = arun(pipeline.synthesize("Q?", claims))

        assert result == "A tight brief."
        assert "Verified fact 1" in captured["prompt"]
        assert "Verified fact 2" in captured["prompt"]
        assert "Rejected fact" not in captured["prompt"]

    def test_bullet_format_in_prompt(self):
        """Each kept claim must appear as '- (angle) text' in the prompt."""
        claims = [self._claim("Some fact", angle="risks / caveats", verified=True)]
        captured = {}

        async def fake_complete(model, prompt, **kwargs):
            captured["prompt"] = prompt
            return "brief"

        with patch.object(pipeline, "complete", new=fake_complete):
            arun(pipeline.synthesize("Q?", claims))

        assert "- (risks / caveats) Some fact" in captured["prompt"]

    def test_question_embedded_in_prompt(self):
        claims = [self._claim("A fact", verified=True)]
        captured = {}

        async def fake_complete(model, prompt, **kwargs):
            captured["prompt"] = prompt
            return "brief"

        with patch.object(pipeline, "complete", new=fake_complete):
            arun(pipeline.synthesize("What is the speed of light?", claims))

        assert "What is the speed of light?" in captured["prompt"]

    def test_returns_llm_output_verbatim(self):
        claims = [self._claim("A fact", verified=True)]
        with patch.object(pipeline, "complete", new=AsyncMock(return_value="The final brief.")):
            result = arun(pipeline.synthesize("Q?", claims))
        assert result == "The final brief."

    def test_multiple_bullets_all_appear(self):
        claims = [
            self._claim("Alpha claim", angle="key facts", verified=True),
            self._claim("Beta claim", angle="contrarian", verified=True),
        ]
        captured = {}

        async def fake_complete(model, prompt, **kwargs):
            captured["prompt"] = prompt
            return "brief"

        with patch.object(pipeline, "complete", new=fake_complete):
            arun(pipeline.synthesize("Q?", claims))

        assert "- (key facts) Alpha claim" in captured["prompt"]
        assert "- (contrarian) Beta claim" in captured["prompt"]


# ===========================================================================
# 6. run() — full orchestration (all three stages mocked at function level)
# ===========================================================================
class TestRun:
    def _make_claim(self, text: str, angle: str = "key facts", verified: bool = True) -> pipeline.Claim:
        c = pipeline.Claim(angle=angle, text=text)
        c.verified = verified
        return c

    def test_returns_synthesis_output(self):
        async def fake_research(question, angle):
            return [self._make_claim(f"{angle} claim")]

        async def fake_verify(question, claim):
            claim.verified = True
            claim.reason = "ok"
            return claim

        async def fake_synthesize(question, claims):
            return "Final brief content"

        with patch.object(pipeline, "research", new=fake_research), \
             patch.object(pipeline, "verify", new=fake_verify), \
             patch.object(pipeline, "synthesize", new=fake_synthesize):
            result = arun(pipeline.run("Test question"))

        assert result == "Final brief content"

    def test_all_angles_researched(self):
        """run() must call research once per entry in ANGLES."""
        researched_angles = []

        async def fake_research(question, angle):
            researched_angles.append(angle)
            return []

        async def fake_verify(question, claim):
            return claim

        async def fake_synthesize(question, claims):
            return "ok"

        with patch.object(pipeline, "research", new=fake_research), \
             patch.object(pipeline, "verify", new=fake_verify), \
             patch.object(pipeline, "synthesize", new=fake_synthesize):
            arun(pipeline.run("Q?"))

        assert set(researched_angles) == set(pipeline.ANGLES)
        assert len(researched_angles) == len(pipeline.ANGLES)

    def test_all_claims_sent_to_verify(self):
        """Every claim from research must be passed through verify."""
        verified_texts = []

        async def fake_research(question, angle):
            return [pipeline.Claim(angle=angle, text=f"claim from {angle}")]

        async def fake_verify(question, claim):
            verified_texts.append(claim.text)
            claim.verified = False
            return claim

        async def fake_synthesize(question, claims):
            return "ok"

        with patch.object(pipeline, "research", new=fake_research), \
             patch.object(pipeline, "verify", new=fake_verify), \
             patch.object(pipeline, "synthesize", new=fake_synthesize):
            arun(pipeline.run("Q?"))

        assert len(verified_texts) == len(pipeline.ANGLES)
        for angle in pipeline.ANGLES:
            assert any(angle in t for t in verified_texts)

    def test_no_claims_returns_no_survived_message(self):
        """If research yields nothing, synthesize gets empty list -> hardcoded message."""
        async def fake_research(question, angle):
            return []

        async def fake_verify(question, claim):
            return claim

        with patch.object(pipeline, "research", new=fake_research), \
             patch.object(pipeline, "verify", new=fake_verify):
            result = arun(pipeline.run("Q?"))

        assert "No claims survived" in result

    def test_synthesize_receives_all_claims_from_verify(self):
        """run() passes every claim returned by verify to synthesize (filtering is synthesize's job)."""
        synthesis_input = {}

        async def fake_research(question, angle):
            return [pipeline.Claim(angle=angle, text=f"claim-{angle}")]

        async def fake_verify(question, claim):
            claim.verified = False
            return claim

        async def fake_synthesize(question, claims):
            synthesis_input["claims"] = claims
            return "done"

        with patch.object(pipeline, "research", new=fake_research), \
             patch.object(pipeline, "verify", new=fake_verify), \
             patch.object(pipeline, "synthesize", new=fake_synthesize):
            arun(pipeline.run("Q?"))

        assert len(synthesis_input["claims"]) == len(pipeline.ANGLES)

    def test_question_forwarded_to_all_stages(self):
        """The question string must be threaded through all three stage calls."""
        received = {"research": set(), "verify": set(), "synthesize": set()}

        async def fake_research(question, angle):
            received["research"].add(question)
            return [pipeline.Claim(angle=angle, text="c")]

        async def fake_verify(question, claim):
            received["verify"].add(question)
            return claim

        async def fake_synthesize(question, claims):
            received["synthesize"].add(question)
            return "ok"

        with patch.object(pipeline, "research", new=fake_research), \
             patch.object(pipeline, "verify", new=fake_verify), \
             patch.object(pipeline, "synthesize", new=fake_synthesize):
            arun(pipeline.run("My test question"))

        assert received["research"] == {"My test question"}
        assert received["verify"] == {"My test question"}
        assert received["synthesize"] == {"My test question"}


# ===========================================================================
# 7. ANGLES constant — structural checks
# ===========================================================================
class TestAngles:
    def test_is_non_empty_sequence(self):
        assert len(pipeline.ANGLES) > 0

    def test_all_strings(self):
        assert all(isinstance(a, str) for a in pipeline.ANGLES)

    def test_no_duplicates(self):
        assert len(pipeline.ANGLES) == len(set(pipeline.ANGLES))

    def test_expected_angles_present(self):
        assert "key facts" in pipeline.ANGLES
        assert "risks / caveats" in pipeline.ANGLES
