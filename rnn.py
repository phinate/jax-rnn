import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from equinox import Module as JaxClass


@jaxtyped
@typechecker
class Parameters(JaxClass):
    embedding_weights: Float[Array, "hidden_state embedding"]
    hidden_state_weights: Float[Array, "hidden_state hidden_state"]
    output_weights: Float[Array, "output hidden_state"]
    hidden_state_bias: Float[Array, "hidden_state"]
    output_bias: Float[Array, "output"]
    embedding_matrix: Float[Array, "embedding vocab"]

e = 3
h = 10
v = 100*e
num_words = 5
o = num_words
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
) -> Float[Array, "output"]:
    return jax.nn.softmax(params.output_weights @ hidden_state + params.output_bias)


@jaxtyped
@typechecker
def loss(
    output: Float[Array, "embedding"],
    next_one_hot_word: Float[Array, "vocab"] 
) -> jax.Array:
    # index the softmax probs at the word of interest
    return -jnp.log(output[next_one_hot_word.astype("bool")])[0]

# @partial(jax.vmap, in_axes = (0, None))
def make_embeddings(
    one_hot_word: Float[Array, "vocab"], 
    params: Parameters
) -> Float[Array, "embedding"]:
    return params.embedding_matrix @ one_hot_vec

embeddings_map = jax.vmap(make_embeddings, in_axes = (0, None))

def rnn(
    data: Float[Array, "sentence vocab"], 
    params: Parameters,
    hidden_size: int = h
) -> Float[Array, "output"]:
    embeddings = embeddings_map(data, params)  # ["sentence embedding"]

    hidden_state = jnp.zeros((hidden_size,))
    outputs = jnp.array([])
    for word in embeddings:
        hidden_state = update_hidden_state(word, hidden_state, params)
        outputs = jnp.concatenate((outputs, output(hidden_state, params)))

    return outputs

    

print(rnn(sentence, params=init_pars).shape)
    