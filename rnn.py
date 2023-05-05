import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, jaxtyped
from beartype import beartype as typechecker
from equinox import Module as JaxClass


@jaxtyped
@typechecker
class Parameters(JaxClass):
    embedding_weights: Float[Array, "hidden_state embedding"]
    hidden_state_weights: Float[Array, "hidden_state hidden_state"]
    output_weights: Float[Array, "vocab hidden_state"]
    hidden_state_bias: Float[Array, "hidden_state"]
    output_bias: Float[Array, "vocab"]
    embedding_matrix: Float[Array, "embedding vocab"]

e = 3
h = 10
v = 100*e
num_words = 5
o = v 
batch_size = 2
sentence = jnp.array([jnp.zeros((v,)).at[3].set(1)]*num_words)
batch = jnp.array([sentence for _ in range(batch_size)])

init_pars = Parameters(
    embedding_weights = jnp.ones((h,e)), 
    hidden_state_weights = jnp.ones((h,h)), 
    output_weights = jnp.ones((o,h)), 
    hidden_state_bias = jnp.ones((h,)), 
    output_bias = jnp.ones((o,)), 
    embedding_matrix = jnp.ones((e,v))
)

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
    output: Float[Array, "embedding"],
    next_one_hot_word: Float[Array, "vocab"] 
) -> Float[Array, ""]:
    # index the softmax probs at the word of interest
    return -jnp.log(output[next_one_hot_word.astype("bool")])[0]

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
    hidden_size: int = h
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
    hidden_size: int = h 
) -> Float[Array, ""]:
    output = rnn(data, params, hidden_size)
    return loss_map(output, next_words)

loss_and_gradient = jax.value_and_grad(forward_pass)

def update_step(
    data: Float[Array, "sentence vocab"],
    next_words: Float[Array, "sentence vocab"], # data shifted by 1 to the right
    params: Parameters,
    learning_rate: float = 4e-2,
    hidden_size: int = h 
) -> Parameters:
    loss_val, gradients = loss_and_gradient(data, next_words, params, hidden_size)
    new_params = params - learning_rate * gradients
    return new_params, loss_val


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

    sentence_length = 5

    vocab_one_hot_indicies = jnp.array([vocab.index(t) for t in all_words], dtype=jnp.int32)
    split_indicies = vocab_one_hot_indicies[:(len(vocab)//sentence_length)*sentence_length].reshape(len(vocab)//sentence_length,sentence_length)

    partition_index = int(2*len(split_indicies/3))
    train = split_indicies[:partition_index]
    test = split_indicies[partition_index:]

    def one_hot_sentence(sentence: Int[Array, "sentence"], vocab_size: int) -> Int[Array, "sentence vocab"]:
        return jnp.array([jnp.zeros((vocab_size,)).at[word].set(1) for word in sentence])
    
    batch_size = 32

    import numpy.random as npr

    def batches(training_data: Array, batch_size: int) -> Generator:
        num_train = training_data[0].shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)

        # batching mechanism, ripped from the JAX docs :)
        def data_stream():
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                    yield [points[batch_idx] for points in train]

        return data_stream()


    batch_iterator = batches(train, batch_size)


