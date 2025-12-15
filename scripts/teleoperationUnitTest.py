import jax
from octo.model.components.action_heads import TeleoperationActionHead
from octo.model.components.base import TokenGroup

# Quick smoke test
action_head = TeleoperationActionHead(
    readout_key="readout_action",
    action_horizon=1,
    action_dim=7,
    use_map=False,
)

# Create dummy inputs
dummy_tokens = jax.numpy.ones((2, 1, 10, 128))  # (batch, window, tokens, embedding)
transformer_outputs = {"readout_action": TokenGroup(tokens=dummy_tokens)}

# Test initialization and forward pass
variables = action_head.init(jax.random.PRNGKey(0), transformer_outputs, train=True)
bound_head = action_head.bind(variables)
mean, cov = bound_head(transformer_outputs, train=True)

print("✓ Action head initializes and runs forward pass")
print(f"  Mean shape: {mean.shape}, Cov shape: {cov.shape}")