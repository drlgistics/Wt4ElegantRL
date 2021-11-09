from ray import tune, init
# from ray.rllib.agents.sac import SACTrainer as Trainer
# from ray.rllib.agents.ddpg import TD3Trainer as Trainer
from ray.rllib.agents.ddpg import ApexDDPGTrainer as Trainer


# from ray.rllib.agents.a3c import A3CTrainer as Trainer
# from ray.rllib.agents.ppo import PPOTrainer as Trainer
# from ray.rllib.agents.ppo import APPOTrainer as Trainer
# from ray.rllib.agents.marwil import MARWILTrainer as Trainer
# from ray.rllib.agents.impala import ImpalaTrainer as Trainer

# from ray.rllib.agents.mbmpo import MBMPOTrainer as Trainer
# from ray.rllib.agents.dreamer import DREAMERTrainer as Trainer
# from ray.rllib.agents.pg import PGTrainer as Trainer
from ray.tune.schedulers.pb2 import PB2
from envs_simple_cta import SimpleCTAEnv
import click

tune.register_env('SimpleCTAEnv',
                  lambda env_config: SimpleCTAEnv(**env_config))


if __name__ == '__main__':
    @click.group()
    def run():
        pass

    @run.command()
    def train():
        pb2 = PB2(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=50000,
            quantile_fraction=0.25,  # copy bottom % with top %
            # Specifies the hyperparam search space
            hyperparam_bounds={
                "lr": [1e-3, 1e-5],
                "gamma": [0.96, 0.99],
            })

        nums_subproc = 5
        nums_gpu = 0.92/(nums_subproc+2)
        config = {
            'env': 'SimpleCTAEnv',
            'env_config': {
                'time_range': (
                    (201901011600, 202101011600),
                    ),
                'slippage': 0,
                'mode': 1
            },
            # 'rollout_fragment_length': 10156,
            'framework': 'torch',
            'num_workers': nums_subproc,
            'num_gpus': nums_gpu,
            'num_gpus_per_worker': nums_gpu,
            'gamma': 0.99,
            'lr': 1e-4,
            'evaluation_interval': 5,
            "evaluation_num_episodes": 5,
            'evaluation_parallel_to_training': False,
            'evaluation_num_workers': 1,

            "evaluation_config": {
                "env_config": {
                    'time_range': (
                        (202101011600, 202106301600),
                        (201701011600, 201706301600),
                        (201706301600, 201801011600),
                        (201801011600, 201806301600),
                        (201806301600, 201901011600),
                        ),
                    'slippage': 0,
                    'mode': 2,
                },
            },
            # 'train_batch_size': 10156,
            "batch_mode": "complete_episodes",
        }

        # training and saving
        analysis = tune.run(
            Trainer,
            stop={
                "timesteps_total": 10156*10000,
                'episode_reward_mean': 1.,
                # 'episode_reward_min': 50,
            },
            # scheduler=pb2,
            # num_samples=nums_subproc,
            config=config,
            keep_checkpoints_num=20,
            checkpoint_freq=5,
            checkpoint_score_attr='episode_reward_mean',
            checkpoint_at_end=True,
            local_dir="./outputs_bt/rllib",
        )
        # retrieve the checkpoint path
        analysis.default_metric = "episode_reward_mean"
        analysis.default_mode = "max"
        checkpoint_path = analysis.get_best_checkpoint(
            trial=analysis.get_best_trial())
        print(f"Trained model saved at {checkpoint_path}")

    @run.command()
    @click.option('--path', '-p', 'path')
    def test(path):

        config = {
            'env': 'SimpleCTAEnv',
            'env_config': {
                'time_start': 202101011600,
                'time_end': 202108311600,
                'slippage': 0,
                'mode': 1
            },
            'framework': 'torch',
            'num_workers': 1,
            'num_gpus': 1,
            'num_gpus_per_worker': 1,
        }
        agent = Trainer(config=config)
        agent.restore(path)
        print(f"Agent loaded from saved model at {path}")

        env = SimpleCTAEnv(**{
            # 'time_start': 201701011600,
            # 'time_end': 201901011600,
            # 'time_start': 202001011600,
            # 'time_end': 202108311600,
            'time_start': 202101011600,
            'time_end': 202110131600,
            # 'time_end': 202110281600,
            'slippage': 0,
            'mode': 2,
        })

        for i in range(10):  # 模拟训练10次
            obs = env.reset()
            done = False
            n = 0
            while not done:
                action = env.action_space.sample()  # 模拟智能体产生动作
                action = agent.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                n += 1
                # print(
                #     # 'action:', action,
                #     # 'obs:', obs,
                #     'reward:', reward,
                #     # 'done:', done
                #     )
            #     break
            # break
            print('第%s次测试完成，执行%s步, 市值%s。' % (i+1, n, env.assets))
        env.close()
        del env

    run()
