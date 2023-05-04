import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
# from typeguard import typechecked as typechecker
from equinox import Module as JaxClass
from functools import partial


@jaxtyped
@typechecker
class Parameters(JaxClass):
    embedding_weights: Float[Array, "hidden_state embedding"]
    hidden_state_weights: Float[Array, "hidden_state hidden_state"]
    output_weights: Float[Array, "embedding hidden_state"]
    hidden_state_bias: Float[Array, "hidden_state"]
    output_bias: Float[Array, "embedding"]

e = 3
h = 10

init_pars = Parameters(jnp.ones((h,e)), jnp.ones((h,h)), jnp.ones((e,h)), jnp.ones((h,)), jnp.ones((e,)))
init_state = jnp.ones((h,))
random_embed = jnp.zeros((e,)).at[3].set(1)  # technically random one-hot, need the embedding matrix...

@jaxtyped
@typechecker
def update_hidden_state(
    embedding: Float[Array, "embedding"], 
    hidden_state: Float[Array, "hidden_state"], 
    params: Parameters
) -> Float[Array, "hidden_state"]:
    return jax.nn.relu(params.hidden_state_weights @ hidden_state + params.embedding_weights @ embedding + params.hidden_state_bias)

update_hidden_state(random_embed, init_state, init_pars)

@jaxtyped
@typechecker
def output(
    hidden_state: Float[Array, "hidden_state"], 
    params: Parameters
) -> Float[Array, "embedding"]:
    return jax.nn.softmax(params.output_weights @ hidden_state + params.output_bias)


@jaxtyped
@typechecker
def loss(
    output: Float[Array, "embedding"],
    next_embedding: Float[Array, "embedding"] 
) -> jax.Array:
    # index the softmax probs at the word of interest
    return -jnp.log(output[next_embedding.astype("bool")])[0]

# @partial(jax.vmap, in_axes = (0, None))
@partial(jax.vmap, in_axes = (0, None))
def make_embeddings(
    one_hot_vec: Float[Array, "vocab"], 
    embedding_matrix: Float[Array, "embedding vocab"]
) -> Float[Array, "embedding"]:
    return embedding_matrix @ one_hot_vec

v = 100*e
E = jnp.ones(shape=(e, v)) + .1
num_words = 5
batch_size = 2
sentence = jnp.array([jnp.zeros((v,)).at[3].set(1)]*num_words)
batch = jnp.array([sentence for _ in range(batch_size)])
print(make_embeddings(batch[0], embedding_matrix=E))

def rnn(data: Float[Array, "batch sentence vocab"]):
    embeddings = make_embeddings(data)  # ["batch sentence embedding"]

    