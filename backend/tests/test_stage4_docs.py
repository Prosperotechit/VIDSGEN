"""The Stage 4A documents say what they must say, honestly, and the Kaggle Support question is short and clear."""

from __future__ import annotations

import re

from app.core.config import REPO_ROOT

DOCS = REPO_ROOT / "docs"
FEAS = (DOCS / "KAGGLE_TUNNEL_FEASIBILITY.md").read_text()
ASK = (DOCS / "KAGGLE_SUPPORT_QUESTION.md").read_text()
SETUP = (DOCS / "KAGGLE_SETUP.md").read_text()


def test_the_support_question_is_short_and_asks_the_right_thing():
    assert len(ASK) < 1800 and len(ASK.split()) < 300
    low = ASK.lower()
    for phrase in ("personal", "non-commercial", "30 minutes", "authenticated", "outbound", "permitted", "terms of use", "acceptable use"):
        assert phrase in low, phrase
    assert "?" in ASK and "draft, not yet sent" in low


def test_the_documents_contain_no_secrets_or_real_hosts():
    for text in (FEAS, ASK):
        assert not re.search(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{16,}|[a-z0-9-]+\.trycloudflare\.com", text)


def test_the_feasibility_document_states_the_policy_position_and_the_gates():
    low = FEAS.lower().replace("**", "")
    assert "not approved" in low and "permission for a public tunnel is not established" in low
    assert "written yes" in low and "no tunnel on kaggle" in low
    for gate in ("kaggle support's written permission", "your explicit approval of stage 4b", "cloudflared_pins.json", "cpu-only"):
        assert gate in low, gate


def test_the_feasibility_document_does_not_overclaim_exactly_once_execution():
    low = FEAS.lower()
    assert "does not guarantee exactly-once execution" in low
    for gap in ("visibility race", "in flight", "judged by id alone", "lost on restart"):
        assert gap in low, gap
    assert "never resubmitted automatically" in low


def test_the_document_covers_the_safeguards_and_the_30_minute_cap():
    low = FEAS.lower()
    assert "off by default" in low and "a job's age alone can never trigger it" in low
    assert "30 minutes" in low and "--max-lifetime-seconds" in low and "--allow-remote" in low


def test_the_dedicated_gpu_alternative_is_documented():
    low = FEAS.lower()
    assert "permitted dedicated gpu environment" in low and "https" in low and "32+ character token" in low
    assert "nothing here selects a provider" in low


def test_the_older_setup_document_now_warns_that_a_tunnel_is_not_approved():
    assert "Public tunnel: NOT approved" in SETUP and "KAGGLE_SUPPORT_QUESTION.md" in SETUP
    assert SETUP.index("Public tunnel: NOT approved") < SETUP.index("### Secure connection (5B)")
