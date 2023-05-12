from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Generator

import jax
import jax.numpy as jnp
from equinox import Module as JaxClass
from jaxtyping import Array, Float, Int


class Parameters(JaxClass):
    embedding_weights: Float[Array, "hidden_state embedding"]
    hidden_state_weights: Float[Array, "hidden_state hidden_state"]
    output_weights: Float[Array, "vocab hidden_state"]
    hidden_state_bias: Float[Array, hidden_state]
    output_bias: Float[Array, vocab]
    embedding_matrix: Float[Array, "embedding vocab"]


def update_hidden_state(
    embedding: Float[Array, embedding],
    hidden_state: Float[Array, hidden_state],
    params: Parameters,
) -> Float[Array, hidden_state]:
    return jax.nn.tanh(
        params.hidden_state_weights @ hidden_state
        + params.embedding_weights @ embedding
        + params.hidden_state_bias
    )


def output(
    hidden_state: Float[Array, hidden_state], params: Parameters
) -> Float[Array, vocab]:
    return jax.nn.softmax(params.output_weights @ hidden_state + params.output_bias)


def loss(
    output: Float[Array, vocab], next_one_hot_word: Float[Array, vocab]
) -> Float[Array, ""]:
    # index the softmax probs at the word of interest
    return -jnp.log(output[jnp.argmax(next_one_hot_word)])


loss_map = jax.vmap(loss, in_axes=(0, 0))


def make_embeddings(
    one_hot_word: Float[Array, vocab], params: Parameters
) -> Float[Array, embedding]:
    return params.embedding_matrix @ one_hot_word


embeddings_map = jax.vmap(make_embeddings, in_axes=(0, None))


def rnn(
    data: Float[Array, "sentence vocab"], params: Parameters, hidden_size: int
) -> Float[Array, "sentence vocab"]:
    embeddings = embeddings_map(data, params)  # ["sentence embedding"]

    hidden_state = jnp.zeros((hidden_size,))

    def update_fn(carry, input):
        hidden_state, params = carry
        word = input
        hidden_state = update_hidden_state(word, hidden_state, params)
        out = output(hidden_state, params)
        return (hidden_state, params), out

    carry = (hidden_state, params)
    _, outputs = jax.lax.scan(update_fn, carry, embeddings)

    return jnp.array(outputs)


def forward_pass(
    data: Float[Array, "sentence vocab"],
    next_words: Float[Array, "sentence vocab"],  # data shifted by 1 to the right
    params: Parameters,
    hidden_size: int,
) -> Float[Array, ""]:
    output = rnn(data, params, hidden_size)
    return loss_map(output, next_words).mean(axis=0)


loss_and_gradient = jax.value_and_grad(forward_pass, argnums=2)
batched_grads = jax.jit(
    jax.vmap(loss_and_gradient, in_axes=(0, 0, None, None)), static_argnums=(3,)
)


def one_hot_sentence(
    sentence: Int[Array, sentence], vocab_size: int
) -> Int[Array, "sentence vocab"]:
    return jnp.array([jnp.zeros((vocab_size,)).at[word].set(1) for word in sentence])


def predict_next_words(
    prompt: str,
    vocab: list[str],
    rnn_params: Parameters,
    rnn_hidden_size: int,
    num_predicted_tokens: int,
    include_prompt=True,
) -> str:
    # Define a regular expression pattern to match all punctuation marks
    punctuation_pattern = r"[^\w\s]"

    # Define a regular expression pattern to match words with apostrophes
    apostrophe_pattern = r"\w+(?:\'\w+)?"
    # Define a regular expression pattern to match newlines
    newline_pattern = r"\n"

    # Combine the three patterns to match all tokens
    token_pattern = (
        punctuation_pattern + "|" + apostrophe_pattern + "|" + newline_pattern
    )

    tokens = re.findall(token_pattern, prompt.lower())
    one_hot_indicies = jnp.array([vocab.index(t) for t in tokens], dtype=jnp.int32)
    sentence = one_hot_sentence(one_hot_indicies, len(vocab))
    embeddings = embeddings_map(sentence, rnn_params)  # ["sentence embedding"]

    hidden_state = jnp.zeros((rnn_hidden_size,))
    outputs = [None] * num_predicted_tokens
    for word in embeddings[:-1]:
        hidden_state = update_hidden_state(word, hidden_state, rnn_params)
    hidden_state = update_hidden_state(embeddings[-1], hidden_state, rnn_params)
    outputs[0] = output(hidden_state, rnn_params)

    for i in range(1, num_predicted_tokens):
        embedded_pred = make_embeddings(outputs[i - 1], rnn_params)
        hidden_state = update_hidden_state(embedded_pred, hidden_state, rnn_params)
        outputs[i] = output(hidden_state, rnn_params)

    res = jnp.array(outputs)
    res_indicies = jnp.argmax(res, axis=1)
    words = [vocab[i] for i in res_indicies]
    out = " ".join(words)
    return prompt + " | " + out if include_prompt else out


if __name__ == "__main__":
    import re

    file_name = "one-fish-two-fish.txt"

    with open(file_name, "r+") as file:
        all_text = file.read()
        # all_text = all_text.replace('\n', ' ').replace('  : ', '')

    # Define a regular expression pattern to match all punctuation marks
    punctuation_pattern = r"[^\w\s]"

    # Define a regular expression pattern to match words with apostrophes
    apostrophe_pattern = r"\w+(?:\'\w+)?"
    # Define a regular expression pattern to match newlines
    newline_pattern = r"\n"

    # Combine the three patterns to match all tokens
    token_pattern = (
        punctuation_pattern + "|" + apostrophe_pattern + "|" + newline_pattern
    )

    # Split the text into tokens, including words with apostrophes as separate tokens
    all_words = re.findall(token_pattern, all_text.lower())
    vocab = list(set(all_words))

    sentence_length = 8  # even for now...

    vocab_one_hot_indicies = jnp.array(
        [vocab.index(t) for t in all_words], dtype=jnp.int32
    )
    split_indicies = vocab_one_hot_indicies[
        : (len(vocab) // sentence_length) * sentence_length
    ].reshape(len(vocab) // sentence_length, sentence_length)
    # make last word random, shouldn't make too much of an impact (could be better handled with special char?)
    split_indicies_labels = jnp.concatenate(
        (
            vocab_one_hot_indicies[
                1 : ((len(vocab) - 1) // sentence_length) * sentence_length
            ],
            jnp.array([0]),
        )
    ).reshape((len(vocab) - 1) // sentence_length, sentence_length)
    partition_index = 6 * int(len(split_indicies) / 7)
    train = split_indicies[:partition_index]
    train_labels = split_indicies_labels[:partition_index]
    valid = split_indicies[partition_index:]
    valid_labels = split_indicies_labels[partition_index:]

    batch_one_hot = jax.vmap(partial(one_hot_sentence, vocab_size=len(vocab)))
    batch_size = 400

    import numpy.random as npr

    def batches(training_data: Array, batch_size: int) -> Generator:
        num_train = training_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)

        # batching mechanism, ripped from the JAX docs :)
        def data_stream():
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                    yield train[batch_idx], train_labels[batch_idx]

        return data_stream()

    batch = batches(train, batch_size)

    e = 30
    h = 16
    v = len(vocab)
    o = v

    pars = Parameters(
        embedding_weights=jax.random.truncated_normal(
            lower=-0.1, upper=0.1, shape=[h, e], key=jax.random.PRNGKey(0)
        ),
        hidden_state_weights=jnp.identity(h),  # keep gradients from exploding
        output_weights=jax.random.truncated_normal(
            lower=-0.1, upper=0.1, shape=[o, h], key=jax.random.PRNGKey(0)
        ),
        hidden_state_bias=jnp.zeros((h,)),  # keep gradients from exploding
        output_bias=jnp.zeros(
            shape=[
                o,
            ]
        ),
        embedding_matrix=jax.random.truncated_normal(
            lower=-0.1, upper=0.1, shape=[e, v], key=jax.random.PRNGKey(0)
        ),
    )
    num_iter = 1000
    lr = 4e-3
    one_hot_valid, one_hot_valid_labels = batch_one_hot(valid), batch_one_hot(
        valid_labels
    )
    best_loss = 999
    best_pars = None

    import optax

    opt = optax.chain(
        optax.clip(1),
        optax.adamw(learning_rate=lr),
    )
    opt_state = opt.init(pars)

    for i in range(num_iter):
        sentences, sentence_labels = next(batch)
        one_hot_sentences, one_hot_sentence_labels = batch_one_hot(
            sentences
        ), batch_one_hot(sentence_labels)
        loss, grads = batched_grads(one_hot_sentences, one_hot_sentence_labels, pars, h)
        valid_loss, _ = batched_grads(one_hot_valid, one_hot_valid_labels, pars, h)
        loss, valid_loss = loss.mean(), valid_loss.mean()
        # pars = jax.tree_map(lambda p, g: p-lr*g.mean(axis=0), pars, grads)
        avg_grads = jax.tree_map(lambda g: g.mean(axis=0), grads)
        updates, opt_state = opt.update(avg_grads, opt_state, params=pars)
        pars = optax.apply_updates(pars, updates)
        if valid_loss < best_loss:
            best_pars = deepcopy(pars)
            best_loss = valid_loss
        if i % 20 == 0:
            print(f"train loss: {loss.mean():.3f}", end=", ")
            print(f"valid loss: {valid_loss.mean():.3f}")

    print(f"best valid loss: {best_loss:.3f}")
    print(predict_next_words("Red fish ", vocab, pars, h, 10, include_prompt=True))
