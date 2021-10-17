from click import command, group, option
from stable_baselines3 import PPO as Trainer
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from envs_simple_cta import SimpleTrainer, SimpleEvaluator

__TRAINER_START__ = 202008311600
__TRAINER_END__ = 202108311600
__TRAINER_STEP__ = 9540

__EVALUATIR_STATR__ = 202005291600
__EVALUATIR_END__ = 202008311600

__BACKTEST_STATR__ = 202108311600
__BACKTEST_END__ = 202110131600

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
                print('action:', action, 'obs:', obs,
                      'reward:', reward, 'done:', done)
            print('第%s次训练完成，执行%s步, 盈亏%s。' % (i+1, n, env.assets))
        env.close()

    @command()
    def test():
        env: SimpleTrainer = SimpleTrainer(
            time_start=__TRAINER_START__,
            time_end=__TRAINER_END__,
        )
        model: Trainer = Trainer('MlpPolicy', env,
                                 learning_rate=0.0001,
                                 # learning_starts=100,
                                 batch_size=128,
                                 # ent_coef='auto_0.1',
                                 gamma=0.99,
                                 verbose=1
                                 )
        model.learn(total_timesteps=__TRAINER_STEP__*2+10, log_interval=1)

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
        checkpoint_callback: CheckpointCallback = CheckpointCallback(
            save_freq=__TRAINER_STEP__*5,
            save_path='./outputs_bt/sb3/',
            name_prefix='SimpleTrainer'
        )

        model: Trainer = Trainer('MlpPolicy', env,
                                 learning_rate=0.0001,
                                 # learning_starts=100,
                                 batch_size=128,
                                #  ent_coef='auto_0.1',
                                 gamma=0.99,
                                #  policy_kwargs=dict(net_arch=[128, 128, 128]),
                                 verbose=1
                                 )
        model.learn(
            total_timesteps=__TRAINER_STEP__*1000+10,
            callback=CallbackList([checkpoint_callback]),
            log_interval=1)
        model.save('SimpleTrainer')

    @command()
    @option('--count', default=10)
    @option('--rtype')
    @option('--name')
    def eval(count, rtype, name):
        rtype = int(rtype)
        if rtype == 2:
            env: SimpleTrainer = SimpleEvaluator(
                time_start=__EVALUATIR_STATR__,
                time_end=__EVALUATIR_END__,
                id=rtype,
            )
        elif rtype == 3:
            env: SimpleTrainer = SimpleEvaluator(
                time_start=__BACKTEST_STATR__,
                time_end=__BACKTEST_END__,
                id=rtype,
            )
        else:
            rtype = 1
            env: SimpleTrainer = SimpleEvaluator(
                time_start=__TRAINER_START__,
                time_end=__TRAINER_END__,
                id=rtype,
            )
        model = Trainer.load('./outputs_bt/sb3/%s' % name)

        for i in range(1, int(count)+1):  # 模拟训练10次
            obs = env.reset()
            done = False
            n = 0
            while not done:
                action = model.predict(obs)  # 模拟智能体产生动作
                obs, reward, done, info = env.step(action[0])
                n += 1
                # print('action:', action, 'obs:', obs, 'reward:', reward, 'done:', done)
            print('第%s次评估完成，执行%s步，奖励%s, 盈亏%s。' % (i, n, reward, env.assets))
        env.close()
        env.analysis()

    run.add_command(debug)
    run.add_command(test)
    run.add_command(train)
    run.add_command(eval)

    run()
