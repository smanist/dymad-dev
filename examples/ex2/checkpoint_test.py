import torch

# Load the checkpoint
checkpoint_path = "./checkpoints/lti_ldm_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, weights_only=False)

# Print all keys
print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

# Print specific values
print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Best Loss: {checkpoint['best_loss']}")
print(f"History length: {len(checkpoint['hist'])}")
print(f"RMSE history length: {len(checkpoint['rmse'])}")

# Print metadata if it exists
if 'metadata' in checkpoint and checkpoint['metadata'] is not None:
    print(f"Metadata keys: {list(checkpoint['metadata'].keys())}")

# Print last few losses from history
if checkpoint['hist']:
    print(f"Last 5 losses: {checkpoint['hist'][-5:]}")