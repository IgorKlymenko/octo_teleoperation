import jax
import jax.numpy as jnp
from octo.model.components.action_heads import TeleoperationActionHead
from octo.model.components.base import TokenGroup

action_head = TeleoperationActionHead(
    readout_key="readout_action",
    action_horizon=1,
    action_dim=7,
    use_map=False,
)

dummy_tokens = jnp.ones((2, 1, 10, 128))        # (batch, window, tokens, embed)
dummy_mask   = jnp.ones((2, 1, 10), dtype=bool) # (batch, window, tokens)

transformer_outputs = {
    "readout_action": TokenGroup(tokens=dummy_tokens, mask=dummy_mask)
}

variables = action_head.init(jax.random.PRNGKey(0), transformer_outputs, train=True)
bound_head = action_head.bind(variables)
mean, cov = bound_head(transformer_outputs, train=True)

print("✓ Action head initializes and runs forward pass")
print(f"  Mean shape: {mean.shape}, Cov shape: {cov.shape}")
