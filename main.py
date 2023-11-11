import functions
from functions import agent, render_it
import os

checkpoint_path = 'checkpoint.pth'

# Check if checkpoint exists and load
if os.path.exists(checkpoint_path):
    total_episodes = agent.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from '{checkpoint_path}'")
else:
    print("No checkpoint found, starting training from scratch")
    total_episodes = 0

# Training loop
for i in range(total_episodes, total_episodes + 100000):

    score, s_list = render_it()
    print(f"Episode {i + 1} has the score of: {score}")

    actor_loss_value = agent.actor_loss_value
    critic_loss_value = agent.critic_loss_value
    functions.agent.write_summary(i, score, actor_loss_value, critic_loss_value)

    # Save checkpoint 
    if (i + 1) % 50 == 0:
        checkpoint_filename = f"checkpoint_{i + 1}.pth"
        agent.save_checkpoint(checkpoint_filename, i + 1)
        print(f"Checkpoint saved to '{checkpoint_filename}'")