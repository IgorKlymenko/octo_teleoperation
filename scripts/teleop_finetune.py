# finetune_teleoperation.py

from octo.model.octo_model import OctoModel
from octo.model.components.action_heads import TeleoperationActionHead
from octo.utils.spec import ModuleSpec
from octo.data.dataset import make_single_dataset

# 1. Load your dataset (contains OBSERVATIONS and ACTIONS, not covariance!)
dataset = make_single_dataset(
    dataset_kwargs=dict(
        name="your_dataset",
        data_dir="./your_data",  # Contains: images, actions, tasks
        # Your dataset has:
        # - observations: images, proprioception, etc.
        # - actions: ground truth actions (what you want to learn from)
        # - tasks: language instructions or goals
    ),
    train=True,
)

# 2. Load pretrained model
pretrained_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

# 3. Replace action head
config = pretrained_model.config.copy()
config["model"]["heads"]["action"] = ModuleSpec.create(
    TeleoperationActionHead,
    action_horizon=1,
    action_dim=7,  # Must match your action space
    readout_key="readout_action",
    use_map=True,
    min_cov_diag=1e-3,
)

# 4. Initialize model
example_batch = next(iter(dataset.batch(1)))
model = OctoModel.from_config(
    config,
    example_batch,
    pretrained_model.text_processor,
    dataset_statistics=dataset.dataset_statistics,
)

# 5. Merge pretrained weights
from octo.utils.train_utils import merge_params
merged_params = merge_params(model.params, pretrained_model.params)
model = model.replace(params=merged_params)

# 6. Training loop
def loss_fn(params, batch, rng, train=True):
    bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
    
    # Step 1: Transformer processes observations → embeddings
    transformer_embeddings = bound_module.octo_transformer(
        batch["observation"],  # ← Your images, proprio, etc.
        batch["task"],         # ← Your language instructions
        batch["observation"]["timestep_pad_mask"],
        train=train,
    )
    
    # Step 2: Action head predicts COVARIANCE from embeddings
    # Step 3: Loss = negative log prob of GROUND TRUTH ACTIONS
    action_loss, action_metrics = bound_module.heads["action"].loss(
        transformer_embeddings,  # ← Used to predict covariance
        batch["action"],         # ← GROUND TRUTH ACTIONS (not covariance!)
        batch["observation"]["timestep_pad_mask"],
        batch["action_pad_mask"],
        train=train,
    )
    
    # The loss function internally:
    # 1. Predicts covariance: cov = action_head(transformer_embeddings)
    # 2. Creates dist = MultivariateNormal(mean=0, cov=cov)
    # 3. Computes loss = -dist.log_prob(batch["action"])
    # 4. Backprop adjusts covariance prediction
    
    return action_loss, action_metrics

# 7. Train!
for batch in dataset:
    loss, metrics = loss_fn(model.params, batch, rng)
    # Backprop adjusts:
    # - Transformer weights (to produce better embeddings)
    # - Action head weights (to predict better covariance)
    # Goal: Make observed actions have high probability under zero-mean Gaussian