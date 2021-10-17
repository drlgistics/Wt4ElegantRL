from click import command, group
from stable_baselines3 import SAC as Trainer
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from envs_simple_cta import SimpleTrainer, SimpleEvaluator


__TRAINER_START__ = 202105311600
__TRAINER_END__ = 2021108311600

__EVALUATIR_STATR_ = 202103311600
__EVALUATIR_END_ = 202105311600

__BACKTEST_STATR_ = 2021108311600
__BACKTEST_END_ = 202110131600

# env = SimpleTrainer(
#     time_start=201909100930,
#     time_end=201912011500,
#     )

# # for i in range(1):  # 模拟训练10次
# #     obs = env.reset()
# #     done = False
# #     n = 0
# #     while not done:
# #         action = env.action_space.sample()  # 模拟智能体产生动作
# #         obs, reward, done, info = env.step(action)
# #         n += 1
# #         print('action:', action, 'obs:', obs, 'reward:', reward, 'done:', done)
# #     print('第%s次训练完成，执行%s步, 盈亏%s。' % (i+1, n, env.assets))
# # env.close()

# model = SAC("MlpPolicy", env, verbose=1)
# model.save("Sac_SimpleCTA")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
if __name__ == '__main__':
    @group()
    def run():
        pass


    @command()
    def debug():
        env: SimpleTrainer = SimpleTrainer(
            time_start=__TRAINER_START__,
            time_end=__TRAINER_END__,
            )

        for i in range(1):  # 模拟训练10次
            obs = env.reset()
            done = False
            n = 0
            while not done:
                action = env.action_space.sample()  # 模拟智能体产生动作
                obs, reward, done, info = env.step(action)
                n += 1
                print('action:', action, 'obs:', obs, 'reward:', reward, 'done:', done)
            print('第%s次训练完成，执行%s步, 盈亏%s。' % (i+1, n, env.assets))
        env.close()


    @command()
    def test():
        env: SimpleTrainer = SimpleTrainer(
            time_start=__TRAINER_START__,
            time_end=__TRAINER_END__,
            )
        model:Trainer = Trainer('MlpPolicy', env, 
                    learning_rate=0.0001, 
                    # learning_starts=100,
                    batch_size=128, 
                    # ent_coef='auto_0.1', 
                    gamma=0.99,
                    verbose=1
                    )
        model.learn(total_timesteps=10000*2+10, log_interval=1)


    @command()
    def train():
        env: SimpleTrainer = SimpleTrainer(
            time_start=__TRAINER_START__,
            time_end=__TRAINER_END__,
            )
        # callback_on_best:StopTrainingOnRewardThreshold = StopTrainingOnRewardThreshold(
        #     reward_threshold=250, 
        #     verbose=1)
        # eval_callback:EvalCallback = EvalCallback(
        #     env, 
        #     best_model_save_path='./outputs_bt/saved/',
        #     log_path='./outputs_bt/saved/', 
        #     callback_on_new_best=callback_on_best, 
        #     verbose=1)
        checkpoint_callback:CheckpointCallback = CheckpointCallback(
            save_freq=100000, 
            save_path='./outputs_bt/saved/',
            name_prefix='SimpleTrainer'
            )

        model:Trainer = Trainer('MlpPolicy', env, 
                    learning_rate=0.0001, 
                    # learning_starts=100,
                    batch_size=128, 
                    ent_coef='auto_0.1', 
                    gamma=0.99,
                    verbose=1
                    )
        model.learn(
            total_timesteps=10000*10000+100, 
            callback=CallbackList([checkpoint_callback]), 
            log_interval=1)
        model.save('SimpleTrainer')

    run.add_command(debug)
    run.add_command(test)
    run.add_command(train)
    
    run()