from click import command, group, option
from elegantrl.agent import AgentModSAC as Agent
from elegantrl.run import Arguments, train_and_evaluate_mp
from envs_simple_cta import SimpleCTAEnv
from gym import make, register
from numpy import inf


class Wt4RLSimpleTrainer(SimpleCTAEnv):
    # env_num = 1
    max_step = 26217
    if_discrete = False

    @property
    def state_dim(self):
        return self.observation_space.shape[0]

    def __init__(self):
        super().__init__(time_start=202001011600, time_end=202108311600)


class Wt4RLSimpleEvaluator(SimpleCTAEnv):
    # env_num = 1
    max_step = 16651
    if_discrete = False

    @property
    def state_dim(self):
        return self.observation_space.shape[0]

    def __init__(self):
        super().__init__(time_start=201901011600, time_end=202001011600)


register('wt4rl-simplecta-trainer-v0', entry_point=Wt4RLSimpleTrainer)
register('wt4rl-simplecta-evaluator-v0', entry_point=Wt4RLSimpleEvaluator)


if __name__ == '__main__':
    @group()
    def run():
        pass

    @command()
    @option('--count', default=1)
    def debug(count):
        env: SimpleCTAEnv = make('wt4rl-simplecta-evaluator-v0')
        print('状态空间', env.observation_space.shape)
        print('动作空间', env.action_space.shape)
        for i in range(1, int(count)+1):  # 模拟训练10次
            obs = env.reset()
            done = False
            n = 0
            while not done:
                action = env.action_space.sample()  # 模拟智能体产生动作
                obs, reward, done, info = env.step(action)
                n += 1
                # print('action:', action, 'obs:', obs, 'reward:', reward, 'done:', done)
            print('第%s次训练完成，执行%s步, 奖励%s, 盈亏%s。' % (i, n, reward, env.assets))
        env.close()

    @command()
    def train():
        args = Arguments(
            env='wt4rl-simplecta-trainer-v0',
            # env='wt4rl-simplecta-evaluator-v0',
            agent=Agent()
        )
        
        #args必须设置的参数
        args.eval_env = 'wt4rl-simplecta-evaluator-v0'
        args.max_step = 26217
        args.state_dim = 550
        args.action_dim = 10
        args.if_discrete = False
        args.target_return = 25  # inf
        # args.if_overwrite = False


        args.break_step = inf
        args.if_allow_break = True


        #
        args.gamma = 0.98
        args.learning_rate = 2 ** -15
        args.worker_num = 1 # 内存小的注意别爆内存

        args.env_num = 1
        args.target_step = args.max_step * 2
        args.learner_gpus = (0,)
        args.workers_gpus = args.learner_gpus
        args.eval_gpu_id = -1
        
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 2
        args.max_memo = 2 ** 20
        # args.repeat_times = 1.5

        #args.net_dim = 2**9
        # args.net_dim = 2 ** 8
        #args.max_memo = 2 ** 22
        # args.break_step = args.max_step*1000
        #args.batch_size = 2 ** 11  # args.net_dim * 2
        # args.repeat_times = 1.5

        # args.eval_gap = 2 ** 9
        # args.eval_times1 = 2 ** 2
        # args.eval_times2 = 2 ** 5

        # args.worker_num = 4
        # args.target_step = args.env.max_step * 1
        # train_and_evaluate(args)

        train_and_evaluate_mp(args)

    @command()
    def test():
        args = Arguments(env=build_env('BipedalWalkerHardcore-v3'), agent=AgentModSAC())
        # args.if_discrete = False
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.max_memo = 2 ** 22
        args.break_step = int(80e6)
        args.batch_size = args.net_dim * 2
        args.repeat_times = 1.5
        args.learning_rate = 2 ** -15
        args.if_per_or_gae = True

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5

        args.worker_num = 1
        args.target_step = args.env.max_step * 1
        args.learner_gpus = (0, )  # single GPU
        train_and_evaluate_mp(args)  # multiple process
        # args = args(
        #     env=build_env('BipedalWalkerHardcore-v3'),
        #     # env='wt4rl-simplecta-evaluator-v0',
        #     agent=AgentModSAC()
        # )
        # #args.eval_env = 'wt4rl-simplecta-evaluator-v0'
        # args.target_step = args.env.max_step
        # args.gamma = 0.98
        # args.net_dim = 2 ** 8
        # args.batch_size = args.net_dim * 2
        # args.learning_rate = 2 ** -15
        # args.repeat_times = 1.5

        # args.max_memo = 2 ** 22
        # args.break_step = 2 ** 24

        # args.eval_gap = 2 ** 8
        # args.eval_times1 = 2 ** 2
        # args.eval_times2 = 2 ** 5

        # args.target_step = args.env.max_step * 1
        # args.worker_num = 1

        # args.learner_gpus = (0,)
        # args.workers_gpus = args.learner_gpus
        # args.visible_gpu = 0
        # args.eval_gpu_id = 0

        # train_and_evaluate_mp(args)

    run.add_command(debug)
    run.add_command(train)
    run.add_command(test)
    # run.add_command(eval)

    run()
