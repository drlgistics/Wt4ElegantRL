from ray.rllib.agents import trainer
from envs_simple_cta import SimpleTrainer, SimpleEvaluator

from ray import tune
from ray.tune.registry import register_env



__TRAINER_START__ = 202105311600
__TRAINER_END__ = 2021108311600

__EVALUATIR_STATR_ = 202103311600
__EVALUATIR_END_ = 202105311600

__BACKTEST_STATR_ = 2021108311600
__BACKTEST_END_ = 202110131600


gpu_nums = 1
worker_nums = 2

def env_creator(env_config):
    return SimpleTrainer(
        time_start=__TRAINER_START__,
        time_end=__TRAINER_END__,
        )
register_env('SimpleTrainer', env_creator)

config={
    'env': 'SimpleTrainer',
    'framework':'torch',
    'num_workers':1,
    'num_gpus':0.2,
    'num_gpus_per_worker':0.015,
    'use_gae':True,
    'gamma':0.99,
    'lr':0.0001,
    'train_batch_size': 128,
    }
stop={
    "timesteps_total": 10000*10000,
    'episode_reward_mean': 50,
    }
analysis = tune.run(
    'PPO',
    name='SimpleTrainer',
    verbose=1,
    num_samples=worker_nums,
    stop=stop,
    metric='episode_reward_mean',
    mode='max',
    keep_checkpoints_num=5,
    checkpoint_freq=1,
    checkpoint_score_attr='episode_reward_mean',
    checkpoint_at_end=True,
    config=config,
    )

# retrieve the checkpoint path
analysis.default_metric = "episode_reward_mean"
analysis.default_mode = "max"
checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial())
print(f"Trained model saved at {checkpoint_path}")