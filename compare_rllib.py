from ray import tune
from ray.rllib.agents.ppo import PPOTrainer as Trainer
# from ray.rllib.agents.sac import SACTrainer as Trainer
from envs_simple_cta import SimpleTrainer, SimpleEvaluator


__TRAINER_START__ = 202105311600
__TRAINER_END__ = 202108311600

__EVALUATIR_STATR__ = 202103311600
__EVALUATIR_END__ = 202105311600

__BACKTEST_STATR__ = 202108311600
__BACKTEST_END__ = 202110131600


def SimpleTrainer_creator(env_config):
    return SimpleTrainer(
        time_start=__TRAINER_START__,
        time_end=__TRAINER_END__,
    )


def SimpleEvaluator_creator(env_config):
    return SimpleEvaluator(
        time_start=__EVALUATIR_STATR__,
        time_end=__EVALUATIR_END__,
    )


tune.register_env('SimpleTrainer', SimpleTrainer_creator)
tune.register_env('SimpleEvaluator', SimpleEvaluator_creator)


config = {
    'env': 'SimpleTrainer',
    'framework': 'torch',
    'num_workers': 1,
    'num_gpus': 0.4,
    'num_gpus_per_worker': 0.5,
    'gamma': 0.98,
    'lr': 5e-6,
    'train_batch_size': 3603,
}


# training and saving
analysis = tune.run(
    Trainer,
    stop={
        "timesteps_total": 3603*10000,
        'episode_reward_mean': 250,
        'episode_reward_min': 50,
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
