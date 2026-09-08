"""Inference-only KV caching adapter for the pinned Auto-AVSR decoder.

The upstream checkpoint/module and beam-search implementation stay unchanged.
Each hypothesis owns immutable self-attention state, so beam pruning, branching
and reordering use the existing scorer select_state contract. Encoder K/V is
shared across all beams of one utterance and reset by init_state.
"""

# Adapts the Apache-2.0 ESPnet/Auto-AVSR decoder computations.
# Upstream decoder: Copyright 2019 Shigeki Karita.
# See third_party/auto_avsr/LICENSE.

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch

from app.config import THIRD_PARTY_ROOT, validate_model_source

validate_model_source()
if str(THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


@dataclass(frozen=True)
class DecoderState:
    # Per-hypothesis tensors: (heads, prefix_length, head_dim).
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    token_log_probabilities: torch.Tensor
    next_log_probabilities: torch.Tensor


def _project(linear, value, attention):
    return (
        linear(value)
        .view(value.shape[0], -1, attention.h, attention.d_k)
        .transpose(1, 2)
    )


def _attend(attention, query, keys, values):
    scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(attention.d_k)
    # There is only one new query; every key is at or before its position.
    return attention.forward_attention(values, scores, None)


class IncrementalDecoderScorer(BatchScorerInterface, torch.nn.Module):
    def __init__(self, decoder):
        torch.nn.Module.__init__(self)
        if (
            len(decoder.embed) != 2
            or not isinstance(decoder.embed[0], torch.nn.Embedding)
            or type(decoder.embed[1]) is not PositionalEncoding
            or decoder.embed[1].reverse
            or not decoder.normalize_before
            or decoder.output_layer is None
            or any(
                not layer.normalize_before or layer.concat_after
                for layer in decoder.decoders
            )
        ):
            raise ValueError(
                "KV caching requires the pinned pre-norm Auto-AVSR decoder configuration."
            )
        self.decoder = decoder
        self._memory_kv = None

    def clear(self):
        self._memory_kv = None

    def init_state(self, x):
        if self.decoder.training or torch.is_grad_enabled():
            raise RuntimeError(
                "KV decoder requires eval() and inference_mode()/no_grad()."
            )
        memory = x.unsqueeze(0)
        self._memory_kv = tuple(
            (
                _project(layer.src_attn.linear_k, memory, layer.src_attn),
                _project(layer.src_attn.linear_v, memory, layer.src_attn),
            )
            for layer in self.decoder.decoders
        )
        return None

    def score(self, ys, state, x):
        scores, states = self.batch_score(ys.unsqueeze(0), [state], x.unsqueeze(0))
        return scores[0], states[0]

    def batch_score(self, ys, states, xs):
        if self.decoder.training or torch.is_grad_enabled():
            raise RuntimeError(
                "KV decoder requires eval() and inference_mode()/no_grad()."
            )
        if self._memory_kv is None:
            raise RuntimeError(
                "Initialize the decoder state for each utterance before scoring."
            )
        batch, length = ys.shape
        if len(states) != batch or any(
            (state is None and length != 1)
            or (state is not None and state.layers[0][0].shape[1] != length - 1)
            for state in states
        ):
            raise ValueError("Decoder cache must match the token prefix length.")
        if any(state is None for state in states) and not all(
            state is None for state in states
        ):
            raise ValueError("Cannot mix initialized and empty decoder states.")

        embedded = self.decoder.embed[0](ys[:, -1:])
        position = self.decoder.embed[1]
        position.extend_pe(embedded[:1].expand(1, length, -1))
        x = position.dropout(
            embedded * position.xscale + position.pe[:, length - 1 : length]
        )
        new_layers = []
        for index, layer in enumerate(self.decoder.decoders):
            residual = x
            normalized = layer.norm1(x)
            attention = layer.self_attn
            query = _project(attention.linear_q, normalized, attention)
            keys = _project(attention.linear_k, normalized, attention)
            values = _project(attention.linear_v, normalized, attention)
            if states[0] is not None:
                keys = torch.cat(
                    (torch.stack([state.layers[index][0] for state in states]), keys),
                    dim=2,
                )
                values = torch.cat(
                    (torch.stack([state.layers[index][1] for state in states]), values),
                    dim=2,
                )
            new_layers.append((keys, values))
            x = residual + layer.dropout(_attend(attention, query, keys, values))
            residual = x
            attention = layer.src_attn
            query = _project(attention.linear_q, layer.norm2(x), attention)
            keys, values = self._memory_kv[index]
            x = residual + layer.dropout(
                _attend(
                    attention,
                    query,
                    keys.expand(batch, -1, -1, -1),
                    values.expand(batch, -1, -1, -1),
                )
            )
            x = x + layer.dropout(layer.feed_forward(layer.norm3(x)))
        logits = self.decoder.output_layer(self.decoder.after_norm(x[:, -1]))
        log_probabilities = torch.log_softmax(logits, dim=-1)

        new_states = []
        for index, previous in enumerate(states):
            history = (
                log_probabilities.new_empty((0,))
                if previous is None
                else torch.cat(
                    (
                        previous.token_log_probabilities,
                        previous.next_log_probabilities[ys[index, -1]].reshape(1),
                    )
                )
            )
            new_states.append(
                DecoderState(
                    layers=tuple(
                        (keys[index], values[index]) for keys, values in new_layers
                    ),
                    token_log_probabilities=history,
                    next_log_probabilities=log_probabilities[index],
                )
            )
        return log_probabilities, new_states

    def token_scores(self, hypothesis, eos: int):
        """Recover decoder-only scores, including the upstream forced-EOS case."""
        state = hypothesis.states["decoder"]
        tokens = hypothesis.yseq[1:]
        history = state.token_log_probabilities
        if len(tokens) == len(history) + 2 and int(tokens[-1]) == eos:
            # Upstream appends an unscored EOS at the maximum decode length.
            tokens = tokens[:-1]
        if len(tokens) != len(history) + 1:
            raise ValueError("Hypothesis tokens and cached probabilities do not align.")
        probabilities = torch.cat(
            (history, state.next_log_probabilities[tokens[-1]].reshape(1))
        )
        return tokens, probabilities
