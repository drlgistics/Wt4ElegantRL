# from stable_baselines3 import SAC as Trainer
from stable_baselines3 import TD3 as Trainer

# from stable_baselines3 import PPO as Trainer
# from stable_baselines3 import A2C as Trainer

from envs_simple_cta import SimpleCTASubProcessEnv, SimpleCTAEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import click


@click.group()
def run():
    pass


@run.command()
def debug():
    learner = SimpleCTASubProcessEnv(**{
        # 'time_start': 202108301600,
        # 'time_end': 202108311600,
        'time_start': 202001011600,
        'time_end': 202108311600,
        'slippage': 0,
        'mode': 1
    })

    evaluator = SimpleCTASubProcessEnv(**{
        # 'time_start': 202108291600,
        # 'time_end': 202108301600,
        'time_start': 201901011600,
        'time_end': 202001011600,
        'slippage': 0,
        'mode': 2,
        'id': 9,
    })

    env = learner

    for i in range(1):  # 模拟训练10次
        obs = env.reset()
        done = False
        n = 0
        while not done:
            action = env.action_space.sample()  # 模拟智能体产生动作
            obs, reward, done, info = env.step(action)
            n += 1
            # print('action:', action, 'obs:', obs,
            #         'reward:', reward, 'done:', done)
        print('第%s次训练完成，执行%s步, 市值%s。' % (i+1, n, env.assets))
    learner.close()


@run.command()
def train():
    learner = SimpleCTASubProcessEnv(**{
        # 'time_start': 202108301600,
        # 'time_end': 202108311600,
        'time_start': 202001011600,
        'time_end': 202108311600,
        'slippage': 0,
        'mode': 1
    })

    evaluator = SimpleCTAEnv(**{
        # 'time_start': 202108291600,
        # 'time_end': 202108301600,
        'time_start': 201901011600,
        'time_end': 202001011600,
        'slippage': 0,
        'mode': 2,
        'id': 9,
    })

    n = 26217

    eval_callback = EvalCallback(
        eval_env=Monitor(evaluator),
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=1200, verbose=1),
        n_eval_episodes=1,
        eval_freq=n*3,
        log_path='./outputs_bt/sb3/',
        best_model_save_path='./outputs_bt/sb3/',
        verbose=1)

    model: Trainer = Trainer('MlpPolicy', learner,
                             learning_rate=2 ** -14, #15: 167, 14:
                            #  learning_rate=1e-4,
                             gamma=0.1 ** (1/12/8),
                             # learning_starts=100,
                             # batch_size=128,
                             # ent_coef='auto_0.1',
                             # policy_kwargs=dict(net_arch=[128, 128, 128]),
                             verbose=1,
                             #  device='cpu',
                             )
    model.learn(
        total_timesteps=n*10000,
        callback=eval_callback,
        log_interval=1
    )
    model.save('SimpleTrainer')


@run.command()
@click.option('--path', '-p', 'path')
def test(path):
    env = SimpleCTAEnv(**{
        # 'time_start': 201701011600,
        # 'time_end': 201901011600,
        # 'time_start': 202001011600,
        # 'time_end': 202108311600,
        'time_start': 202108311600,
        'time_end': 202110131600,
        # 'time_end': 202110281600,
        'slippage': 0,
        'mode': 2,
        'id': 2,
    })
    model = Trainer.load(path)

    for i in range(1):  # 模拟训练10次
        obs = env.reset()
        done = False
        n = 0
        while not done:
            action = model.predict(obs)[0]
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


if __name__ == '__main__':
    run()
