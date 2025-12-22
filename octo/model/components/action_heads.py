from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Tuple

import distrax
from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from octo.model.components.base import TokenGroup
from octo.model.components.diffusion import cosine_beta_schedule, create_diffusion_model
from octo.model.components.tokenizers import BinTokenizer
from octo.model.components.transformer import MAPHead
from octo.model.components.unet import ConditionalUnet1D, unet_squaredcos_cap_v2
from octo.utils.typing import PRNGKey

import tensorflow_probability.substrates.jax as tfp


class ActionHead(ABC):
    """Action prediction modules that take in the transformer token outputs and predict actions.

    Each action head here does chunked action prediction: i.e. at every timestep, it tries to predict the next
    `action_horizon` actions into the future from that timestep.  Setting `action_horizon=1` corresponds to
    the typical action prediction setup.
    """

    @abstractmethod
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        train: bool = False,
        embodiment_action_dim: Optional[int] = None,
    ) -> Array:
        """Predict the action for the last timestep in the window. Returns shape
        (*sample_shape, batch_size, action_horizon, action_dim).
        """
        raise NotImplementedError


def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)


def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
    }


def discrete_loss(
    discrete_tokenizer: BinTokenizer,
    logits: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
) -> Array:
    """
    Args:
        discrete_tokenizer: BinTokenizer to use on ground_truth_value
        logits: shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in w/ shape (batch_dims...)
        mask: broadcastable to ground_truth_value
    """
    labels = discrete_tokenizer(ground_truth_value)
    labels_one_hot = jax.nn.one_hot(labels, logits.shape[-1])

    logprobs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(logprobs * labels_one_hot, axis=-1)
    loss = masked_mean(loss, mask)

    # compute accuracy between predicted actions and target actions
    pred_label = jnp.argmax(logits, axis=-1)
    accuracy = pred_label == labels
    accuracy = masked_mean(accuracy, mask)

    # detokenize the predicted actions
    pred_value = discrete_tokenizer.decode(pred_label)
    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
        "accuracy": accuracy,
    }


class ContinuousActionHead(nn.Module, ActionHead):
    """Predicts continuous actions (as opposed to discretized).

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action], and then
    optimized using a standard regression loss.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    use_map: bool = False
    action_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0
    loss_type: str = "mse"

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()
        self.mean_proj = nn.Dense(self.action_horizon * self.action_dim)

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train: bool = True
    ) -> jax.Array:
        """
        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, action_horizon, action_dim)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        mean = self.mean_proj(embeddings)
        mean = rearrange(
            mean, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
        )
        mean = jnp.tanh(mean / self.max_action) * self.max_action
        return mean

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the action regression objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension

        Returns:
            loss: float
            metrics: dict
        """
        # (batch, window_size, action_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        # combine the timestep pad mask with the action pad mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask

        loss, metrics = continuous_loss(mean, actions, mask, loss_type=self.loss_type)
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        mean = self(transformer_outputs, train=train)[:, -1]
        return jnp.broadcast_to(mean, sample_shape + mean.shape)


class DiscreteActionHead(nn.Module, ActionHead):
    """
    A basic action decoding head that predicts discretized actions using the transformer token embeddings.


    self.token_per determines how many tokens are used to represent each action.
        - If "" (an empty string): then a single token is responsible for producing the action logits
            for all dimensions at all future prediction horizons.
        - If "action_horizon", then we use `self.action_horizon` tokens, each responsible for producing the action logits
            for all dimensions at the corresponding future prediction horizon.
        - If "action_dim_and_action_horizon", then we use `self.action_horizon * self.action_dim` tokens, where
            each token is responsible for the logits for the specific dim and timestep.

    If multi-head attention pooling is used (use_map=True), then the correct number of tokens is automatically
    created, otherwise readout_key must have exactly the right number of tokens.
    """

    readout_key: str
    use_map: bool = False
    token_per: str = "action_dim_and_action_horizon"
    action_horizon: int = 1
    action_dim: int = 7
    vocab_size: int = 256
    normalization_type: str = "uniform"

    def setup(self):
        total_output = self.action_horizon * self.action_dim * self.vocab_size

        if self.token_per == "":
            self.n_tokens = 1
            self.final_layer_size = total_output
        elif self.token_per == "action_horizon":
            self.n_tokens = self.action_horizon
            self.final_layer_size = total_output // self.action_horizon
        elif self.token_per == "action_dim_and_action_horizon":
            self.n_tokens = self.action_horizon * self.action_dim
            self.final_layer_size = self.vocab_size
        else:
            raise ValueError(f"Invalid token_per: {self.token_per}")

        if self.use_map:
            self.map_head = MAPHead(num_readouts=self.n_tokens)

        self.vocab_proj = nn.Dense(self.final_layer_size)
        self.action_tokenizer = BinTokenizer(
            n_bins=self.vocab_size,
            bin_type=self.normalization_type,
        )

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train: bool = True
    ) -> jax.Array:
        """
        Returns:
            logits: array w/ shape (batch_size, window_size, action_horizon, action_dim, vocab_size)
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:
            embeddings = self.map_head(token_group, train=train)
        else:
            embeddings = token_group.tokens
            assert (
                embeddings.shape[-2] == self.n_tokens
            ), f"Discrete action head expects {self.n_tokens} tokens"

        # Now, embeddings is (batch_size, window_size, n_tokens, embedding_size)
        batch_size, window_size = embeddings.shape[:2]

        logits = self.vocab_proj(embeddings)
        logits = logits.reshape(
            batch_size,
            window_size,
            self.action_horizon,
            self.action_dim,
            self.vocab_size,
        )
        return logits

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ):
        """Computes the loss for the discretized action objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension

        Returns:
            loss: float
            metrics: dict
        """
        # get the logits for all the actions by taking the action tokens of each timestep,
        # unfolding the action_horizon dim, and projecting to the vocab size
        # (batch, window_size, action_horizon, action_dim, token_embedding_size)
        action_logits = self(transformer_outputs, train=train)

        # combine the timestep pad mask with the action pad mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask

        loss, metrics = discrete_loss(
            self.action_tokenizer, action_logits, actions, mask
        )

        # For MSE, sum over action dimension instead of averaging
        metrics["mse"] = metrics["mse"] * self.action_dim

        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = True,
        argmax: bool = False,
        sample_shape: tuple = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        **unused_kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        action_logits = self(transformer_outputs, train=train)[:, -1]

        if argmax:
            action_tokens = jnp.argmax(action_logits, axis=-1).astype(jnp.int32)
            action_tokens = jnp.broadcast_to(
                action_tokens, sample_shape + action_tokens.shape
            )
        else:
            dist = distrax.Categorical(logits=action_logits / temperature)
            action_tokens = dist.sample(seed=rng, sample_shape=sample_shape).astype(
                jnp.int32
            )
        return self.action_tokenizer.decode(action_tokens)


class MSEActionHead(ContinuousActionHead):
    max_action: float = 5.0
    loss_type: str = "mse"
    use_map: bool = True


class L1ActionHead(ContinuousActionHead):
    max_action: float = 5.0
    loss_type: str = "l1"
    use_map: bool = True


class TokenPerDimActionHead(DiscreteActionHead):
    token_per: str = "action_dim_and_action_horizon"


class DiffusionActionHead(nn.Module):
    """Predicts actions uses a diffusion process.

    Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
    actions are then predicted using a diffusion process conditioned on this embedding. The diffusion model
    architecture is an MLP with residual connections (see `octo.model.components.diffusion`).

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    use_map: bool = False
    action_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0
    loss_type: str = "mse"

    # diffusion-specific config with sane defaults
    time_dim: int = 32
    num_blocks: int = 3
    dropout_rate: float = 0.0
    hidden_dim: int = 256
    use_layer_norm: bool = True
    diffusion_steps: int = 20
    n_diffusion_samples: int = 1

    def setup(self):
        if self.use_map:
            self.map_head = MAPHead()

        # create the diffusion model (score network)
        self.diffusion_model = create_diffusion_model(
            self.action_dim * self.action_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

        # create beta schedule
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.cumprod(self.alphas)

    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Optional[ArrayLike] = None,
        noisy_actions: Optional[ArrayLike] = None,
        train: bool = True,
    ) -> jax.Array:
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        # time and noisy_actions are None during initialization, so we replace them with a dummy array
        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError(
                "Must provide time and noisy_actions when calling diffusion action head"
            )
        elif self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*embeddings.shape[:2], self.action_dim * self.action_horizon),
                dtype=jnp.float32,
            )
        pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
        return pred_eps

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the diffusion objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension

        Returns:
            loss: float
            metrics: dict
        """
        batch_size, window_size = timestep_pad_mask.shape

        # fold action_dim and action_horizon into one dimension
        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        # piggy-back on the dropout rng chain for diffusion rng
        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
        time = jax.random.randint(
            time_key,
            (self.n_diffusion_samples, batch_size, window_size, 1),
            0,
            self.diffusion_steps,
        )
        noise = jax.random.normal(
            noise_key, (self.n_diffusion_samples,) + actions_flat.shape
        )

        scale = jnp.sqrt(self.alpha_hats[time])
        std = jnp.sqrt(1 - self.alpha_hats[time])
        noisy_actions = scale * actions_flat[None] + std * noise

        pred_eps = self(
            transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
        )

        # combine the timestep pad `mask with the action pad mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask
        # flatten the mask to match the flat actions
        mask = rearrange(mask, "b w h a -> b w (h a)")
        # add a dimension to the mask for n_diffusion_samples
        mask = mask[None]

        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type=self.loss_type)
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        embodiment_action_dim: Optional[int] = None,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        if embodiment_action_dim is None:
            logging.warning(
                "embodiment_action_dim is highly recommended for diffusion action head"
                " if any action dimensions were masked during training"
            )
        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
        module, variables = self.unbind()

        action_mask = jnp.ones(
            (
                *sample_shape,
                batch_size,
                window_size,
                self.action_horizon,
                self.action_dim,
            ),
            dtype=bool,
        )
        if embodiment_action_dim is not None:
            action_mask = action_mask.at[..., embodiment_action_dim:].set(False)
        flat_action_mask = rearrange(action_mask, "... p a -> ... (p a)")

        def scan_fn(carry, time):
            current_x, rng = carry
            input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))

            eps_pred = module.apply(
                variables, transformer_outputs, input_time, current_x, train=train
            )

            alpha_1 = 1 / jnp.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=current_x.shape)
            current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)

            current_x = jnp.clip(current_x, -self.max_action, self.max_action)

            # set non-eval actions to the noise that would have been seen during training
            current_x = jnp.where(
                flat_action_mask, current_x, jnp.sqrt(1 - self.alpha_hats[time]) * z
            )

            return (current_x, rng), ()

        rng, key = jax.random.split(rng)
        noise = jax.random.normal(
            key,
            (
                *sample_shape,
                batch_size,
                window_size,
                self.action_horizon * self.action_dim,
            ),
        )

        (actions_flat, _), () = jax.lax.scan(
            scan_fn,
            (noise, rng),
            jnp.arange(self.diffusion_steps - 1, -1, -1),
        )

        actions = rearrange(
            actions_flat,
            "... (h a) -> ... h a",
            h=self.action_horizon,
            a=self.action_dim,
        )
        # only get the last timestep in the window
        return actions[..., -1, :, :]


class UNetDDPMActionHead(nn.Module):
    """Predicts actions using a diffusion process and a U-Net architecture (unlike MLP above)

    Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
    actions are then predicted using a diffusion process conditioned on this embedding. The diffusion model
    architecture is an 1D unet based on the implementation from Chi et al: https://arxiv.org/abs/2303.04137

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    action_dim: int
    action_horizon: int

    use_map: bool = (False,)
    flatten_tokens: bool = (False,)
    timesteps: int = 100
    max_action: float = 1.0
    clip_sample: Optional[float] = None
    variance_type: str = "fixed_large"

    def setup(self):
        self.action_proj = nn.Dense(self.action_dim)
        betas = unet_squaredcos_cap_v2(self.timesteps).astype(jnp.float32)
        self.alphas = 1.0 - betas  # So betas = 1 - alphas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)

        self.model = ConditionalUnet1D(
            down_features=(256, 512, 1024),
            mid_layers=2,
            time_features=128,
            kernel_size=5,
        )

    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Optional[ArrayLike] = None,
        noisy_actions: Optional[ArrayLike] = None,
        train: bool = True,
    ) -> jax.Array:
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )

        if self.use_map:  # Multi-head attention pooling
            assert not self.flatten_tokens, "Cannot use MAP token and flattening!"
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        elif self.flatten_tokens:  # concatenate tokens in final dim
            embeddings = token_group.tokens.reshape((*token_group.tokens.shape[:2], -1))
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        # time and noisy_actions are None during initialization, so we replace them with a dummy array
        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError(
                "Must provide time and noisy_actions when calling diffusion action head"
            )
        elif self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*embeddings.shape[:2], self.action_horizon, self.action_dim),
                dtype=jnp.float32,
            )  # (b, w, p, a)
        pred_eps = self.model(embeddings, action=noisy_actions, time=time, train=train)
        pred_eps = self.action_proj(pred_eps)
        return pred_eps

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        action_pad_mask: ArrayLike,
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the diffusion objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + action_horizon - 1, action_dim)
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        batch_size, window_size = timestep_pad_mask.shape[:2]

        actions = jnp.clip(actions, -self.max_action, self.max_action)

        # piggy-back on the dropout rng chain for diffusion rng
        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
        time = jax.random.randint(
            time_key,
            (batch_size, window_size, 1),
            0,
            self.timesteps,
        )
        noise = jax.random.normal(noise_key, actions.shape)

        # Add noise to the action according to the schedule
        sqrt_alpha_prod = jnp.sqrt(self.alphas_cumprod[time[:, None]])  # (B, 1, 1)
        sqrt_one_minus_alpha_prod = jnp.sqrt(
            1 - self.alphas_cumprod[time[:, None]]
        )  # (B, 1, 1)
        noisy_actions = sqrt_alpha_prod * actions + sqrt_one_minus_alpha_prod * noise

        pred_eps = self(
            transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
        )

        # combine the timestep-level pad mask with the action-dimension-level pad mask
        mask = (
            jnp.broadcast_to(action_pad_mask[:, None, None, :], actions.shape)
            * timestep_pad_mask
        )

        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type="mse")
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        embodiment_action_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> jax.Array:
        """
        Code inspired by diffusers:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm_flax.py
        """
        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
        module, variables = self.unbind()

        action_mask = jnp.ones(
            (
                batch_size,
                window_size,
                self.action_horizon,
                self.action_dim,
            ),
            dtype=bool,
        )

        if embodiment_action_dim is not None:
            action_mask = action_mask.at[..., embodiment_action_dim:].set(False)
        else:
            logging.warning(
                "embodiment_action_dim is highly recommended for diffusion action head"
                " if any action dimensions were masked during training"
            )

        def loop_body(i, args):
            sample, rng = args
            time = self.timesteps - 1 - i
            # Note that here time is (B, 1, 1) where as in loss in is (B, 1)
            time = jnp.broadcast_to(time, (sample.shape[0], 1, 1))
            alpha = self.alphas[time]
            alpha_prod_t = self.alphas_cumprod[time]
            alpha_prod_t_prev = jnp.where(
                time > 0,
                self.alphas_cumprod[time - 1],
                jnp.array(1.0, dtype=jnp.float32),
            )

            # Run the model. Reduce time to (B, 1) for the model.
            eps = module.apply(
                variables,
                transformer_outputs,
                time=time,
                noisy_actions=sample,
                train=train,
            )

            # Predict x_0, clip if desired.
            orig = (sample - jnp.sqrt(1 - alpha_prod_t) * eps) / jnp.sqrt(alpha_prod_t)
            if self.clip_sample is not None:
                orig = jnp.clip(orig, -self.clip_sample, self.clip_sample)

            # Compute x_{t-1} using x_0
            orig_coeff = jnp.sqrt(alpha_prod_t_prev) * (1 - alpha) / (1 - alpha_prod_t)
            current_coeff = (
                jnp.sqrt(alpha) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            )

            prev = orig_coeff * orig + current_coeff * sample

            # Add noise according to the schedule
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha)
            if self.variance_type == "fixed_large":
                variance = 1 - alpha
            elif self.variance_type == "fixed_small":
                variance = jnp.clip(variance, a_min=1e-20)
            else:
                raise ValueError("Invalid schedule provided")

            rng, key = jax.random.split(rng)
            variance = jnp.where(
                time > 0, variance, jnp.zeros(eps.shape, dtype=jnp.float32)
            )
            z = jax.random.normal(key, shape=sample.shape, dtype=jnp.float32)
            prev = prev + jnp.sqrt(variance) * z

            # set non-eval actions to the noise that would have been seen during training
            prev = jnp.where(action_mask, prev, jnp.sqrt(1 - alpha_prod_t) * z)

            return (prev, rng)

        rng, key = jax.random.split(rng)
        noisy_action = jax.random.normal(
            key,
            (
                batch_size,
                window_size,
                self.action_horizon,
                self.action_dim,
            ),
        )

        noisy_action, _ = jax.lax.fori_loop(
            0, self.timesteps, loop_body, (noisy_action, rng)
        )

        return noisy_action

class TeleoperationActionHead(nn.Module, ActionHead):
    """Predicts actions using a teleoperation model with Gaussian distribution.

    This action head predicts a Gaussian distribution over actions with zero mean and
    a single learned covariance matrix that applies to all actions in the horizon.
    It uses the negative log probability of the ground truth actions under this zero-mean
    Gaussian distribution as the loss function. This allows the model to capture uncertainty
    (variance) in action predictions while assuming actions are centered at zero.

    The covariance matrix is predicted once per timestep and shared across all action_horizon
    steps, making it more parameter-efficient than predicting separate covariances for each
    horizon step.

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using
    multi-head attention pooling (use_map=True). It is recommended to use MAP when decoding
    from the observation token stream.
    """

    readout_key: str
    use_map: bool = False
    action_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0
    # Minimum value for diagonal covariance to ensure numerical stability
    min_cov_diag: float = 1e-3

    def setup(self):
        """Initialize the action head components.
        
        Creates:
        - MAP head for attention pooling (if use_map=True)
        - Dense layer to predict action covariance matrix parameters
        Note: Mean is assumed to be zero, so no mean prediction layer is needed.
        """
        if self.use_map:
            self.map_head = MAPHead()
        
        # Predict parameters for a single covariance matrix that applies to all actions in the horizon
        # (mean is assumed to be zero)
        # We'll predict a lower triangular matrix and construct the covariance from it
        # Number of parameters in lower triangular matrix: action_dim * (action_dim + 1) / 2
        # This single covariance matrix will be shared across all action_horizon steps
        n_cov_params = self.action_dim * (self.action_dim + 1) // 2
        self.cov_proj = nn.Dense(n_cov_params)

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Predicts a single covariance matrix for a Gaussian distribution over actions (mean is zero).
        The same covariance matrix is shared across all actions in the horizon.
        
        Args:
            transformer_outputs: Dictionary containing token groups from transformer
            train: Whether in training mode
            
        Returns:
            mean: Zero mean w/ shape (batch_size, window_size, action_horizon, action_dim)
            cov: Predicted action covariance matrices w/ shape 
                 (batch_size, window_size, action_horizon, action_dim, action_dim)
                 Note: The same covariance matrix is broadcast across all action_horizon steps
        """
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        
        # Extract embeddings from transformer outputs
        if self.use_map:  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # (batch_size, window_size, embedding_size)

        # Mean is assumed to be zero
        batch_size, window_size = embeddings.shape[:2]
        mean = jnp.zeros((batch_size, window_size, self.action_horizon, self.action_dim))

        # Predict covariance matrix parameters (single prediction for entire horizon)
        # We predict a lower triangular matrix L such that cov = L @ L.T
        # This ensures the covariance matrix is positive semi-definite
        cov_params_flat = self.cov_proj(embeddings)
        # cov_params_flat shape: (batch_size, window_size, n_cov_params)
        
        # Construct lower triangular matrix from parameters
        # We use a parameterization where we predict the lower triangular part
        L = jnp.zeros((batch_size, window_size, self.action_dim, self.action_dim))
        idx = 0
        for i in range(self.action_dim):
            for j in range(i + 1):
                L = L.at[:, :, i, j].set(cov_params_flat[:, :, idx])
                idx += 1
        
        # Construct single covariance matrix: cov = L @ L.T + min_cov_diag * I
        # This ensures positive definiteness and numerical stability
        cov_single = jnp.matmul(L, jnp.transpose(L, (0, 1, 3, 2)))
        cov_single = cov_single + self.min_cov_diag * jnp.eye(self.action_dim)[None, None, :, :]
        # cov_single shape: (batch_size, window_size, action_dim, action_dim)
        
        # Broadcast the single covariance matrix across all action_horizon steps
        cov = jnp.broadcast_to(
            cov_single[:, :, None, :, :],
            (batch_size, window_size, self.action_horizon, self.action_dim, self.action_dim)
        )
        # Final shape: (batch_size, window_size, action_horizon, action_dim, action_dim)

        return mean, cov

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the negative log probability loss for the zero-mean Gaussian action distribution.

        The loss is the negative log probability of the ground truth actions under the
        predicted zero-mean Gaussian distribution. This allows the model to learn the
        uncertainty (covariance) of actions while assuming the mean is zero.

        Args:
            transformer_outputs: must contain self.readout_key with shape 
                (batch_size, window_size, num_tokens, embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the 
                timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the 
                action dimension is not a padding dimension

        Returns:
            loss: scalar loss value (negative log probability, masked and averaged)
            metrics: dict containing loss, mse (mean squared error of actions from zero), and other metrics
        """
        # Predict covariance of action distribution (mean is zero)
        # mean: (batch_size, window_size, action_horizon, action_dim) - all zeros
        # cov: (batch_size, window_size, action_horizon, action_dim, action_dim)
        mean, cov = self(transformer_outputs, train=train)

        # Combine the timestep pad mask with the action pad mask
        # mask: (batch_size, window_size, action_horizon, action_dim)
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask

        # Compute negative log probability for each action dimension
        # We'll compute log probability per timestep and action dimension
        batch_size, window_size, action_horizon, action_dim = actions.shape
        
        # Reshape for per-timestep, per-horizon computation
        # Flatten batch and window dimensions for easier processing
        actions_flat = actions.reshape(-1, action_horizon, action_dim)
        mean_flat = mean.reshape(-1, action_horizon, action_dim)
        cov_flat = cov.reshape(-1, action_horizon, action_dim, action_dim)
        mask_flat = mask.reshape(-1, action_horizon, action_dim)
        
        # Compute log probabilities for each (batch*window, horizon) pair
        total_log_probs = []
        for h in range(action_horizon):
            # Get actions, mean (zero), and cov for this horizon
            actions_h = actions_flat[:, h, :]  # (batch*window, action_dim)
            mean_h = mean_flat[:, h, :]  # (batch*window, action_dim) - all zeros
            cov_h = cov_flat[:, h, :, :]  # (batch*window, action_dim, action_dim)
            mask_h = mask_flat[:, h, :]  # (batch*window, action_dim)
            
            # Create multivariate normal distribution with zero mean for each sample
            # We need to handle masking: only compute log prob for valid action dimensions
            # For simplicity, we'll compute log prob for all dimensions and then mask
            dist = tfp.distributions.MultivariateNormalFullCovariance(
                loc=mean_h, covariance_matrix=cov_h
            )
            log_probs_h = dist.log_prob(actions_h)  # (batch*window,)
            
            # Apply mask: only consider timesteps where at least one action dim is valid
            valid_mask = mask_h.any(axis=-1)  # (batch*window,)
            log_probs_h = jnp.where(valid_mask, log_probs_h, 0.0)
            total_log_probs.append(log_probs_h)
        
        # Stack log probabilities across horizons
        # Shape: (batch*window, action_horizon)
        log_probs = jnp.stack(total_log_probs, axis=-1)
        
        # Average across action horizons (each horizon contributes equally)
        log_probs = jnp.mean(log_probs, axis=-1)  # (batch*window,)
        
        # Reshape back to (batch_size, window_size)
        log_probs = log_probs.reshape(batch_size, window_size)
        
        # Apply timestep mask
        log_probs = jnp.where(timestep_pad_mask, log_probs, 0.0)
        
        # Compute negative log probability as loss
        # We sum over valid timesteps and divide by number of valid timesteps
        n_valid = jnp.sum(timestep_pad_mask)
        loss = -jnp.sum(log_probs) / jnp.clip(n_valid, a_min=1.0)
        
        # Also compute MSE for monitoring (mean squared error of actions from zero)
        # Since mean is zero, this measures how far actions are from zero
        mse = jnp.square(actions)  # actions - 0 = actions
        mse = masked_mean(mse, mask)
        
        # Scale by action_dim to match other action heads
        loss = loss * self.action_dim
        mse = mse * self.action_dim
        
        metrics = {
            "loss": loss,
            "mse": mse,
            "nll": loss,  # Negative log likelihood (same as loss)
        }
        
        return loss, metrics


    ## Do not actually need, but is here
    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        train: bool = False,
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        embodiment_action_dim: Optional[int] = None,
    ) -> Array:
        """Predict actions by sampling from the predicted zero-mean Gaussian distribution.

        For the last timestep in the window, samples actions from the predicted
        zero-mean Gaussian distribution. If argmax=True, returns zero (the mean/mode).

        Args:
            transformer_outputs: Dictionary containing token groups from transformer
            train: Whether in training mode
            argmax: If True, return zero (the mean/mode) instead of sampling
            sample_shape: Shape for additional samples (e.g., (n_samples,) for n_samples samples)
            rng: Random number generator key for sampling
            temperature: Temperature for sampling (scales the covariance)
            embodiment_action_dim: Optional action dimension for masking

        Returns:
            actions: Sampled actions or zeros w/ shape (*sample_shape, batch_size, action_horizon, action_dim)
        """
        # Predict covariance for the last timestep (mean is zero)
        mean, cov = self(transformer_outputs, train=train)
        # Get only the last timestep
        mean_last = mean[:, -1]  # (batch_size, action_horizon, action_dim) - all zeros
        cov_last = cov[:, -1]  # (batch_size, action_horizon, action_dim, action_dim)

        if argmax:
            # Return zero (the mean/mode of the zero-mean distribution)
            actions = jnp.broadcast_to(mean_last, sample_shape + mean_last.shape)
        else:
            # Sample from the zero-mean Gaussian distribution
            if rng is None:
                raise ValueError("rng must be provided for sampling")
            
            # Scale covariance by temperature
            cov_scaled = cov_last * (temperature ** 2)
            
            # Sample for each horizon
            batch_size = mean_last.shape[0]
            samples = []
            rng_current = rng
            for h in range(self.action_horizon):
                mean_h = mean_last[:, h, :]  # (batch_size, action_dim) - all zeros
                cov_h = cov_scaled[:, h, :, :]  # (batch_size, action_dim, action_dim)
                
                # Create zero-mean distribution and sample
                dist = tfp.distributions.MultivariateNormalFullCovariance(
                    loc=mean_h, covariance_matrix=cov_h
                )
                # Split RNG for this horizon
                rng_current, rng_h = jax.random.split(rng_current)
                sample_h = dist.sample(seed=rng_h, sample_shape=sample_shape)
                # sample_h shape: (*sample_shape, batch_size, action_dim)
                samples.append(sample_h)
            
            # Stack along action_horizon dimension
            # Shape: (*sample_shape, batch_size, action_horizon, action_dim)
            actions = jnp.stack(samples, axis=-2)
        
        return actions