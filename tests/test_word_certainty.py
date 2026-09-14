from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from app.model import MpsSpatialMaxPool, group_word_certainties


def test_sentencepiece_tokens_are_grouped_by_word() -> None:
    token_list = ["<blank>", "▁HEL", "LO", "▁WORLD", "<eos>"]
    result = group_word_certainties(
        [1, 2, 3, 4],
        [math.log(0.81), math.log(0.49), math.log(0.64), math.log(0.99)],
        token_list,
    )
    assert [item.word for item in result] == ["HELLO", "WORLD"]
    assert result[0].certainty == pytest.approx(math.sqrt(0.81 * 0.49))
    assert result[0].token_count == 2
    assert result[1].certainty == pytest.approx(0.64)


def test_grouping_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        group_word_certainties([1], [], ["<blank>", "▁WORD"])


def test_mps_spatial_pool_matches_the_original_model_pool() -> None:
    video = torch.randn(2, 4, 5, 12, 14)

    expected = F.max_pool3d(
        video,
        kernel_size=(1, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )
    actual = MpsSpatialMaxPool()(video)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", [(1, 4, 7, 13, 15), (2, 4, 7, 14, 16)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_pool_preserves_odd_shapes_and_noncontiguous_inputs(shape, noncontiguous):
    video = torch.randn(shape)
    if noncontiguous:
        video = video.transpose(-1, -2)
    expected = F.max_pool3d(video, (1, 3, 3), (1, 2, 2), (0, 1, 1))
    assert torch.equal(MpsSpatialMaxPool()(video), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_pool_matches_cpu_reference_on_mps():
    video = torch.randn(1, 4, 7, 13, 15)
    expected = F.max_pool3d(video, (1, 3, 3), (1, 2, 2), (0, 1, 1))
    assert torch.equal(MpsSpatialMaxPool()(video.to("mps")).cpu(), expected)
