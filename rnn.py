from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype as typechecker
from equinox import Module as JaxClass
from typing import Generator
from functools import partial


@jaxtyped
@typechecker
class Parameters(JaxClass):
    embedding_weights: Float[Array, "hidden_state embedding"]
    hidden_state_weights: Float[Array, "hidden_state hidden_state"]
    output_weights: Float[Array, "vocab hidden_state"]
    hidden_state_bias: Float[Array, "hidden_state"]
    output_bias: Float[Array, "vocab"]
    embedding_matrix: Float[Array, "embedding vocab"]


@jaxtyped
@typechecker
def update_hidden_state(
    embedding: Float[Array, "embedding"], 
    hidden_state: Float[Array, "hidden_state"], 
    params: Parameters
) -> Float[Array, "hidden_state"]:
    return jax.nn.relu(params.hidden_state_weights @ hidden_state + params.embedding_weights @ embedding + params.hidden_state_bias)


@jaxtyped
@typechecker
def output(
    hidden_state: Float[Array, "hidden_state"], 
    params: Parameters
) -> Float[Array, "vocab"]:
    return jax.nn.softmax(params.output_weights @ hidden_state + params.output_bias)


@jaxtyped
@typechecker
def loss(
    output: Float[Array, "vocab"],
    next_one_hot_word: Float[Array, "vocab"] 
) -> Float[Array, ""]:
    # index the softmax probs at the word of interest
    return -jnp.log(output[jnp.argmax(next_one_hot_word)])

loss_map = jax.vmap(loss, in_axes=(0, 0))

@jaxtyped
@typechecker
def make_embeddings(
    one_hot_word: Float[Array, "vocab"], 
    params: Parameters
) -> Float[Array, "embedding"]:
    return params.embedding_matrix @ one_hot_word

embeddings_map = jax.vmap(make_embeddings, in_axes = (0, None))

@jaxtyped
@typechecker
def rnn(
    data: Float[Array, "sentence vocab"], 
    params: Parameters,
    hidden_size: int
) -> Float[Array, "sentence vocab"]:
    embeddings = embeddings_map(data, params)  # ["sentence embedding"]

    hidden_state = jnp.zeros((hidden_size,))
    outputs = []
    for word in embeddings:
        hidden_state = update_hidden_state(word, hidden_state, params)
        outputs.append(output(hidden_state, params))

    return jnp.array(outputs)

# @jax.jit
@jaxtyped
@typechecker
def forward_pass(
    data: Float[Array, "sentence vocab"],
    next_words: Float[Array, "sentence vocab"], # data shifted by 1 to the right
    params: Parameters,
    hidden_size: int
) -> Float[Array, ""]:
    output = rnn(data, params, hidden_size)
    return loss_map(output, next_words).mean(axis=0)

batched_forward_pass = jax.vmap(forward_pass, in_axes=(0, 0, None, None))
loss_and_gradient = jax.value_and_grad(forward_pass, argnums=2)

@jaxtyped
@typechecker
def update_step(
    data: Float[Array, "sentence vocab"],
    next_words: Float[Array, "sentence vocab"], # data shifted by 1 to the right
    params: Parameters,
    hidden_size: int,
) -> tuple[Parameters, Float[Array, ""]]:
    loss_val, gradients = loss_and_gradient(data, next_words, params, hidden_size)

    return gradients, loss_val

batched_grads = jax.jit(jax.vmap(update_step, in_axes=(0, 0, None, None)),static_argnums=(3,))

if __name__ == "__main__":

    import re
    file_name = 'bee-movie-names.txt'

    with open(file_name, 'r+') as file:
        all_text = file.read()
        all_text = all_text.replace('\n', ' ').replace('  : ', '')
        
    # Define a regular expression pattern to match all punctuation marks
    punctuation_pattern = r'[^\w\s]'

    # Define a regular expression pattern to match words with apostrophes
    apostrophe_pattern = r'\w+(?:\'\w+)?'

    # Combine the two patterns to match all tokens
    token_pattern = punctuation_pattern + '|' + apostrophe_pattern

    # Split the text into tokens, including words with apostrophes as separate tokens
    all_words = re.findall(token_pattern, all_text.lower())
    vocab = list(set(all_words))

    sentence_length = 2

    vocab_one_hot_indicies = jnp.array([vocab.index(t) for t in all_words], dtype=jnp.int32)
    split_indicies = vocab_one_hot_indicies[:(len(vocab)//sentence_length)*sentence_length].reshape(len(vocab)//sentence_length,sentence_length)
    # make last word random, shouldn't make too much of an impact (could be better handled with special char?)
    split_indicies_labels = jnp.concatenate((vocab_one_hot_indicies[1:((len(vocab)-1)//sentence_length)*sentence_length], jnp.array([0]))).reshape((len(vocab)-1)//sentence_length,sentence_length)
    partition_index = 2*int(len(split_indicies)/3)
    train = split_indicies[:partition_index]
    train_labels = split_indicies_labels[:partition_index]
    valid = split_indicies[partition_index:]
    valid_labels = split_indicies_labels[partition_index:]

    def one_hot_sentence(sentence: Int[Array, "sentence"], vocab_size: int) -> Int[Array, "sentence vocab"]:
        return jnp.array([jnp.zeros((vocab_size,)).at[word].set(1) for word in sentence])
    
    batch_one_hot = jax.vmap(partial(one_hot_sentence, vocab_size = len(vocab)))
    batch_size = 32

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
    h = 4
    v = len(vocab)
    o = v 

    pars = Parameters(
        embedding_weights = jnp.ones((h,e)), 
        hidden_state_weights = jnp.ones((h,h)), 
        output_weights = jnp.ones((o,h)), 
        hidden_state_bias = jnp.ones((h,)), 
        output_bias = jnp.ones((o,)), 
        embedding_matrix = jnp.ones((e,v))
    )
    num_iter = 100
    lr = 3e-4
    # init pars
    one_hot_valid, one_hot_valid_labels = batch_one_hot(valid), batch_one_hot(valid_labels)
    print(one_hot_valid_labels.shape)
    for i in range(num_iter):
        sentences, sentence_labels = next(batch)
        one_hot_sentences, one_hot_sentence_labels = batch_one_hot(sentences), batch_one_hot(sentence_labels)
        grads, loss = batched_grads(one_hot_sentences, one_hot_sentence_labels, pars, h)
        _, valid_loss = batched_grads(one_hot_valid, one_hot_valid_labels, pars, h)
        scaled_avg_gradients = jax.tree_map(lambda g: lr * g.mean(axis=0), grads)
        pars = jax.tree_map(lambda x, y: x-y, pars, scaled_avg_gradients)
        if i % 10 ==0:
            print(f"train loss: {loss.mean():.3f}")
            print(f"valid loss: {valid_loss.mean():.3f}")