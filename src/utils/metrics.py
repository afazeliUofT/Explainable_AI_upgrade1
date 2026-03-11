"""Metrics for bit/block error rates."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class ErrorCounts:
    n_bits: int
    n_bit_errors: int
    n_blocks: int
    n_block_errors: int

    def ber(self) -> float:
        return float(self.n_bit_errors) / max(1, int(self.n_bits))

    def bler(self) -> float:
        return float(self.n_block_errors) / max(1, int(self.n_blocks))


def count_errors(b: tf.Tensor, b_hat: tf.Tensor) -> ErrorCounts:
    """Count BER/BLER for tensors shaped [B, num_tx, num_bits]."""
    b = tf.cast(b, tf.int32)
    b_hat = tf.cast(b_hat, tf.int32)


        # Prevent silent broadcasting (which can yield BER>1 and invalid BLER).
    rb = b.shape.rank
    rh = b_hat.shape.rank
    if rb is None:
        rb = int(tf.rank(b).numpy())
    if rh is None:
        rh = int(tf.rank(b_hat).numpy())

    # Common case: transmitter is [B,1,N] but some decoders return [B,N]
    if (rb == 3) and (rh == 2):
        b_hat = b_hat[:, None, :]
    elif (rb == 2) and (rh == 3):
        b = b[:, None, :]

    tf.debugging.assert_equal(
        tf.shape(b),
        tf.shape(b_hat),
        message="count_errors: b and b_hat shapes mismatch (would broadcast and corrupt BER/BLER).",
    )

    bit_err = tf.not_equal(b, b_hat)
    n_bit_errors = int(tf.reduce_sum(tf.cast(bit_err, tf.int32)).numpy())
    n_bits = int(tf.size(b).numpy())

    # Block error if any bit differs
    block_err = tf.reduce_any(bit_err, axis=-1)  # [B, num_tx]
    n_block_errors = int(tf.reduce_sum(tf.cast(block_err, tf.int32)).numpy())
    n_blocks = int(tf.size(block_err).numpy())

    return ErrorCounts(
        n_bits=n_bits,
        n_bit_errors=n_bit_errors,
        n_blocks=n_blocks,
        n_block_errors=n_block_errors,
    )
