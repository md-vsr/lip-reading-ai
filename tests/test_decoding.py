"""Equivalence tests use real upstream modules, not downloaded model weights."""

from types import SimpleNamespace as NS

import pytest
import torch

from app.decoding import IncrementalDecoderScorer
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.scorers.ctc import CTCPrefixScorer


def decoder(seed=0, layers=2):
    torch.manual_seed(seed)
    return TransformerDecoder(
        odim=12,
        attention_dim=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=layers,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.1,
        src_attention_dropout_rate=0.1,
    ).eval()


@pytest.mark.parametrize("seed", [0, 1, 2])
@torch.inference_mode()
def test_cache_matches_reference_after_beam_branching_and_reordering(seed):
    original = decoder(seed)
    cached = IncrementalDecoderScorer(original).eval()
    memory = torch.randn(20, 8)
    cached.init_state(memory)
    ys = torch.tensor([[11]])
    reference_states = [None]
    cached_states = [None]
    for parents, tokens in [([0, 0], [2, 3]), ([1, 0, 1], [5, 2, 6]), ([2, 0], [8, 7])]:
        xs = memory.expand(len(ys), -1, -1)
        expected, reference_states = original.batch_score(ys, reference_states, xs)
        actual, cached_states = cached.batch_score(ys, cached_states, xs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        ys = torch.cat((ys[parents], torch.tensor(tokens).unsqueeze(1)), dim=1)
        reference_states = [reference_states[parent] for parent in parents]
        cached_states = [cached_states[parent] for parent in parents]
    expected, _ = original.batch_score(
        ys, reference_states, memory.expand(len(ys), -1, -1)
    )
    actual, _ = cached.batch_score(ys, cached_states, memory.expand(len(ys), -1, -1))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("forced_eos", [False, True])
@torch.inference_mode()
def test_cached_word_scores_match_teacher_forcing_including_forced_eos(forced_eos):
    original = decoder()
    cached = IncrementalDecoderScorer(original).eval()
    memory = torch.randn(15, 8)
    cached.init_state(memory)
    prefix = torch.tensor([11, 2, 4, 3])
    state = None
    for length in range(1, len(prefix) + 1):
        _, state = cached.score(prefix[:length], state, memory)
    yseq = torch.cat((prefix, torch.tensor([5, 11] if forced_eos else [11])))
    hypothesis = NS(yseq=yseq, states={"decoder": state})
    ids, actual = cached.token_scores(hypothesis, eos=11)
    scored_sequence = yseq[:-1] if forced_eos else yseq
    mask = torch.ones(
        1, len(scored_sequence) - 1, len(scored_sequence) - 1, dtype=torch.bool
    ).tril()
    logits, _ = original(
        scored_sequence[:-1].unsqueeze(0), mask, memory.unsqueeze(0), None
    )
    expected = (
        logits[0].log_softmax(-1).gather(1, scored_sequence[1:].unsqueeze(1)).squeeze(1)
    )
    assert torch.equal(ids, scored_sequence[1:])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_cache_projects_each_source_once_and_only_one_new_self_token():
    original = decoder(layers=1)
    cached = IncrementalDecoderScorer(original).eval()
    calls = {name: [] for name in ("self_k", "self_v", "cross_k", "cross_v")}
    layer = original.decoders[0]
    handles = []
    for name, projection in [
        ("self_k", layer.self_attn.linear_k),
        ("self_v", layer.self_attn.linear_v),
        ("cross_k", layer.src_attn.linear_k),
        ("cross_v", layer.src_attn.linear_v),
    ]:
        handles.append(
            projection.register_forward_pre_hook(
                lambda module, args, name=name: calls[name].append(args[0].shape[1])
            )
        )
    memory = torch.randn(20, 8)
    cached.init_state(memory)
    state = None
    for length in (1, 2, 3):
        _, state = cached.score(torch.ones(length, dtype=torch.long), state, memory)
    for handle in handles:
        handle.remove()
    assert calls == {
        "self_k": [1, 1, 1],
        "self_v": [1, 1, 1],
        "cross_k": [20],
        "cross_v": [20],
    }


@pytest.mark.parametrize("beam_size", [1, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
@torch.inference_mode()
def test_joint_ctc_attention_beam_hypotheses_match_reference(beam_size, seed):
    original = decoder(seed)
    cached = IncrementalDecoderScorer(original).eval()
    ctc = CTC(12, 8, 0.1).eval()
    memory = torch.randn(10, 8)

    def beam(scorer):
        return BatchBeamSearch(
            scorers={"decoder": scorer, "ctc": CTCPrefixScorer(ctc, eos=11)},
            weights={"decoder": 0.9, "ctc": 0.1},
            beam_size=beam_size,
            vocab_size=12,
            sos=11,
            eos=11,
            pre_beam_score_key="decoder",
        ).eval()

    expected = beam(original)(memory)
    actual = beam(cached)(memory)
    assert len(actual) == len(expected) and actual
    for a, b in zip(actual, expected):
        assert torch.equal(a.yseq, b.yseq)
        torch.testing.assert_close(a.score, b.score, rtol=1e-5, atol=1e-5)
        ids, scores = cached.token_scores(a, eos=11)
        assert len(ids) == len(scores)
    # No source K/V may leak into the next clip on the same scorer.
    new_memory = torch.randn(8, 8)
    expected_next = beam(original)(new_memory)
    actual_next = beam(cached)(new_memory)
    assert torch.equal(actual_next[0].yseq, expected_next[0].yseq)


def test_cache_refuses_training_or_grad_enabled_use():
    cached = IncrementalDecoderScorer(decoder())
    with pytest.raises(RuntimeError, match="inference_mode"):
        cached.init_state(torch.randn(10, 8))


@torch.inference_mode()
def test_cache_refuses_mismatched_prefix_length():
    cached = IncrementalDecoderScorer(decoder())
    memory = torch.randn(10, 8)
    cached.init_state(memory)
    with pytest.raises(ValueError, match="prefix length"):
        cached.score(torch.tensor([11, 1]), None, memory)
