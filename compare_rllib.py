from ray import tune
# from ray.rllib.agents.ppo import PPOTrainer as Trainer
# from ray.rllib.agents.ppo import APPOTrainer as Trainer
from ray.rllib.agents.sac import SACTrainer as Trainer
# from ray.rllib.agents.ddpg import TD3Trainer as Trainer
from envs_simple_cta import SimpleCTAEnv


tune.register_env('SimpleCTAEnv',
                  lambda env_config: SimpleCTAEnv(**env_config))


config = {
    'env': 'SimpleCTAEnv',
    'env_config': {'time_start': 202001011600, 'time_end': 202108311600, 'mode': 1},
    'rollout_fragment_length': 26217,
    'framework': 'torch',
    'num_workers': 1,
    'num_gpus': 0.3,
    'num_gpus_per_worker': 0.3,
    # 'gamma': 0.1 ** (1/12/8),
    # 'lr': 2 ** -14,
    'simple_optimizer': True,
    'evaluation_interval': 10,
    "evaluation_num_episodes": 5,
    'evaluation_parallel_to_training': False,
    'evaluation_num_workers': 1,

    "evaluation_config": {
        "env_config": {
            'time_start': 201901011600, 'time_end': 202001011600, 'mode': 2
        },
    },
    'train_batch_size': 26217,
}


# training and saving
analysis = tune.run(
    Trainer,
    stop={
        "timesteps_total": 26217*10000,
        'episode_reward_mean': 2.,
        # 'episode_reward_min': 50,
    },
    config=config,
    keep_checkpoints_num=5,
    checkpoint_freq=1,
    checkpoint_score_attr='episode_reward_mean',
    checkpoint_at_end=True,
    local_dir="./outputs_bt/rllib",
)
# retrieve the checkpoint path
analysis.default_metric = "episode_reward_mean"
analysis.default_mode = "max"
checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial())
print(f"Trained model saved at {checkpoint_path}")

# # load and restore model
# agent = ppo.PPOTrainer(env=env_name)
# agent.restore(checkpoint_path)
# print(f"Agent loaded from saved model at {checkpoint_path}")

# # inference
# env = gym.make(env_name)
# obs = env.reset()
# for i in range(1000):
#     action = agent.compute_single_action(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         print(f"Cart pole dropped after {i} steps.")
#         break
