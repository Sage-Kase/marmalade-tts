"""Tests for the per-fragment stream planner (marmalade_tts.stream_session).

Everything here is offline: the fake engines phonemize by prefixing each
word with a stress mark, so phoneme strings stay readable and no model is
involved. What is being pinned is the fragment-boundary contract from the
ratified streaming spec: the ramp offset carries, the run gap is graded,
conditioning never crosses a fragment seam, and style rows come from the
fragment (never from a whole utterance the daemon cannot see yet).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from marmalade_tts import chunking
from marmalade_tts.stream_session import (RenderItem, ends_sentence,
                                          fragment_gap_ms, plan_fragment)


def _ph(text: str) -> str:
    """The fake phonemizer: one stressed unit per word, punctuation kept."""
    return " ".join("ˈ" + w for w in text.split())


class FakePhEngine:
    """A PHONEME_STREAM engine (kokoro's shape) with no model."""

    MAX_CHARS = 500
    PHONEME_STREAM = True
    STYLE_ROWS = "ph-utterance"
    model_size = "test"

    def __init__(self):
        self.ph_calls = []
        self.text_calls = []

    def phonemize(self, text, **kw):
        return _ph(text)

    def synthesize_phonemes(self, ph_text, out_path, **kw):
        self.ph_calls.append((ph_text, kw))

    def synthesize(self, text, out_path, **kw):
        self.text_calls.append((text, kw))


class FakeTextEngine:
    """No phoneme path — the text-space fallback."""

    MAX_CHARS = 500
    model_size = "test"

    def __init__(self):
        self.text_calls = []

    def synthesize(self, text, out_path, **kw):
        self.text_calls.append((text, kw))


class FakeClauseEngine(FakePhEngine):
    """kitten's shape: phoneme renders, but planned in text space with
    graded clause gaps."""

    TEXT_CLAUSE_PLAN = True
    STYLE_ROWS = "text-sentence"


LONG = ("the market square filled with people carrying baskets and the "
        "bells rang out over the rooftops until dusk")


def _plan(engine, text, **kw):
    return plan_fragment(engine, text, engine_name="fake", eng_cfg={},
                         ph_key="test:ph", **kw)


# ── the ramp offset carries across fragments ────────────────────────────────

class TestRampOffset:
    def test_later_fragments_use_larger_chunk_targets(self):
        eng = FakePhEngine()
        eng.STREAM_BANDS = (("fast", float("inf"), (20, 400), 2, 2),)
        first = _plan(eng, LONG + ".", step=0)
        later = _plan(eng, LONG + ".", step=1)
        # Step 0 pays the small opening target; one chunk later the ramp
        # has grown enough to keep the whole sentence in one render.
        assert len(first.items) >= 2
        assert len(first.items[0].text) <= 30
        assert len(later.items) == 1

    def test_next_step_advances_by_the_items_planned(self):
        eng = FakePhEngine()
        eng.STREAM_BANDS = (("fast", float("inf"), (20, 400), 2, 2),)
        p = _plan(eng, LONG + ".", step=2)
        assert p.next_step == 2 + len(p.items)

    def test_ramp_offset_carries_on_the_text_path_too(self):
        eng = FakeTextEngine()
        text = ("we left before dawn; we travelled light; we told nobody; "
                "the road was empty; the sky stayed dark.")
        first = _plan(eng, text, step=0)
        later = _plan(eng, text, step=4)
        assert len(later.items[0].text) > len(first.items[0].text)

    def test_empty_fragment_plans_nothing_and_holds_the_step(self):
        p = _plan(FakePhEngine(), "   \n ", step=3)
        assert p.items == []
        assert p.next_step == 3


# ── conditioning stops at the fragment seam ─────────────────────────────────

class TestConditioning:
    def test_no_context_or_lookahead_on_the_fragment_edges(self):
        eng = FakePhEngine()
        eng.STREAM_BANDS = (("fast", float("inf"), (20, 30, 40), 2, 2),)
        # A mid-utterance fragment: nothing may leak in from the previous
        # fragment, and nothing may reach forward into the next one.
        p = _plan(eng, LONG + ".", step=1)
        assert len(p.items) >= 2
        assert p.items[0].context is None
        assert p.items[-1].lookahead is None

    def test_conditioning_inside_a_fragment_is_kept(self):
        eng = FakePhEngine()
        eng.STREAM_BANDS = (("fast", float("inf"), (20, 30, 40), 2, 2),)
        p = _plan(eng, LONG + ".", step=0)
        assert len(p.items) >= 3
        # Interior pieces of one sentence run still condition both ways —
        # that is what keeps a mid-sentence cut from sounding restarted.
        assert p.items[1].context
        assert p.items[1].lookahead

    def test_two_fragments_never_share_conditioning(self):
        eng = FakePhEngine()
        a = _plan(eng, "The first sentence lands here.", step=0)
        b = _plan(eng, "The second sentence follows it.",
                  step=a.next_step, sentence=1)
        assert a.items[-1].lookahead is None
        assert b.items[0].context is None


# ── style rows ──────────────────────────────────────────────────────────────

class TestStyleRows:
    def test_ph_utterance_engine_gets_per_fragment_rows_midstream(self):
        # kokoro's stock rule indexes the pack by the WHOLE utterance's
        # phoneme length — unavailable mid-stream. The ratified spec says
        # per-fragment phoneme length (ph-sentence shape) instead.
        eng = FakePhEngine()
        p = _plan(eng, "One one one.", step=0)
        assert [it.style_ref for it in p.items] == [len(_ph("One one one.")) - 1]

    def test_rows_are_per_run_inside_a_multi_sentence_fragment(self):
        eng = FakePhEngine()
        p = _plan(eng, "One one one. Two two two two.", step=0)
        runs = [_ph("One one one."), _ph("Two two two two.")]
        assert [it.style_ref for it in p.items] == [len(r) - 1 for r in runs]

    def test_text_row_engines_index_by_text_length(self):
        eng = FakeClauseEngine()
        p = _plan(eng, "One one one.", step=0)
        assert [it.style_ref for it in p.items] == [len("One one one.")]


# ── the run gap between fragments ───────────────────────────────────────────

class TestFragmentGap:
    def test_sentence_end_gets_the_run_gap(self):
        p = _plan(FakePhEngine(), "One one one.", step=0)
        assert p.items[-1].gap_after_ms == chunking.RUN_GAP_MS

    def test_mid_sentence_fragment_gets_the_small_clause_gap(self):
        p = _plan(FakePhEngine(), "One one one,", step=0)
        assert p.items[-1].gap_after_ms == chunking.CLAUSE_GAP_MS

    def test_graded_engines_use_their_own_sentence_gap(self):
        eng = FakeClauseEngine()
        assert (_plan(eng, "One one one.", step=0).items[-1].gap_after_ms
                == chunking.CLAUSE_SENT_GAP_MS)
        assert (_plan(eng, "One one one,", step=0).items[-1].gap_after_ms
                == chunking.CLAUSE_GAP_MS)

    def test_every_fragment_carries_a_trailing_gap(self):
        # The daemon can't know a fragment is the last one until `end`
        # arrives, so the last chunk keeps a trailing pause — inaudible,
        # and accepted by the ratified spec.
        p = _plan(FakePhEngine(), "The only fragment.", step=0)
        assert p.items[-1].gap_after_ms > 0

    def test_internal_run_gaps_survive(self):
        p = _plan(FakePhEngine(), "One one one. Two two two.", step=0)
        assert p.items[0].gap_after_ms == chunking.RUN_GAP_MS

    @pytest.mark.parametrize("text,expected", [
        ("Done.", True), ('He said "stop!"', True), ("A question?", True),
        ("これは。", True), ("a list item,", False), ("no mark", False),
        ("(parenthetical)", False), ("Ends here.)", True),
    ])
    def test_ends_sentence(self, text, expected):
        assert ends_sentence(text) is expected

    def test_gap_helper_matches_the_planned_gap(self):
        eng = FakePhEngine()
        assert fragment_gap_ms(eng, "Yes.") == chunking.RUN_GAP_MS
        assert fragment_gap_ms(FakeClauseEngine(), "Yes.") == \
            chunking.CLAUSE_SENT_GAP_MS


# ── the text-space fallback ─────────────────────────────────────────────────

class TestTextFallback:
    def test_non_phoneme_engine_plans_in_text_space(self):
        p = _plan(FakeTextEngine(), "One one one. Two two two.", step=0)
        assert all(not it.phonemes for it in p.items)
        assert "".join(it.text for it in p.items).replace(" ", "") == \
            "Oneoneone.Twotwotwo."

    def test_phonemize_failure_falls_back_to_text(self):
        eng = FakePhEngine()

        def boom(text, **kw):
            raise RuntimeError("daemon down")
        eng.phonemize = boom
        p = _plan(eng, "One one one. Two two two.", step=0)
        assert p.items and all(not it.phonemes for it in p.items)

    def test_chunking_disabled_renders_the_fragment_whole(self):
        eng = FakeTextEngine()
        eng.MAX_CHARS = None
        p = _plan(eng, "One one one. Two two two.", step=0)
        assert [it.text for it in p.items] == ["One one one. Two two two."]


# ── RenderItem ──────────────────────────────────────────────────────────────

class TestRenderItem:
    def test_gate_item_prices_conditioning_and_gap(self):
        it = RenderItem("abcde", context="xy", lookahead="z",
                        gap_after_ms=150)
        assert it.cond_chars == 3
        assert it.gate_item() == (5, 8, 0.15)

    def test_phoneme_items_render_through_synthesize_phonemes(self, tmp_path):
        eng = FakePhEngine()
        p = _plan(eng, "One one one.", step=0)
        p.items[0].synthesize(eng, str(tmp_path / "a.wav"), speed=1.0)
        ph_text, kw = eng.ph_calls[0]
        assert ph_text == _ph("One one one.")
        assert kw["style_ref"] == len(_ph("One one one.")) - 1
        assert kw["pad_marks"]           # mark pauses floored, as in stream_play
        assert kw["speed"] == 1.0
        assert eng.text_calls == []

    def test_clause_plans_skip_pad_marks(self, tmp_path):
        # A clause mark is always chunk-final there and the graded gap
        # already carries its pause — padding would double it.
        eng = FakeClauseEngine()
        p = _plan(eng, "One one one.", step=0)
        p.items[0].synthesize(eng, str(tmp_path / "a.wav"))
        assert eng.ph_calls[0][1]["pad_marks"] is None

    def test_text_items_pass_conditioning_only_when_advertised(self,
                                                               tmp_path):
        eng = FakeTextEngine()
        p = _plan(eng, "One one one. Two two two.", step=0)
        p.items[0].synthesize(eng, str(tmp_path / "a.wav"))
        assert "context" not in eng.text_calls[0][1]
        assert "lookahead" not in eng.text_calls[0][1]

        eng2 = FakeTextEngine()
        eng2.STREAM_CONTEXT = True
        eng2.STREAM_LOOKAHEAD = True
        p2 = _plan(eng2, "We left before dawn; we travelled light; we told "
                         "nobody at all; the road stayed empty.", step=0)
        assert len(p2.items) >= 2
        assert p2.items[1].context and p2.items[0].lookahead
        assert p2.items[0].context is None
        assert p2.items[-1].lookahead is None


# ── wire bookkeeping ────────────────────────────────────────────────────────

class TestSentenceStamping:
    def test_every_item_carries_the_fragment_index_and_text(self):
        eng = FakePhEngine()
        eng.STREAM_BANDS = (("fast", float("inf"), (20, 30, 40), 2, 2),)
        p = _plan(eng, LONG + ".", step=0, sentence=4)
        assert len(p.items) >= 2
        assert {it.sentence for it in p.items} == {4}
        assert {it.sentence_text for it in p.items} == {LONG + "."}

    def test_sentence_text_can_be_the_pre_preprocessing_text(self):
        # The daemon plans the PROCESSED text but reports the RAW fragment
        # the client sent — that is what the client files as spoken.
        eng = FakePhEngine()
        p = _plan(eng, "Twenty two dogs.", sentence=0,
                  sentence_text="22 dogs.")
        assert p.items[0].sentence_text == "22 dogs."
