from click import command, group, option
# from elegantrl.agent import AgentPPO as Agent
# from elegantrl.agent import AgentSAC as Agent
# from elegantrl.agent import AgentModSAC as Agent
from elegantrl.agent import AgentTD3 as Agent
from elegantrl.run import Arguments, train_and_evaluate
from envs_simple_cta import SimpleCTASubProcessEnv
from gym import make, register
from numpy import inf


class Wt4RLSimpleTrainer(SimpleCTASubProcessEnv):
    env_num = 1
    max_step = 26217
    if_discrete = False

    @property
    def state_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

    def __init__(self):
        super().__init__(**{
        # 'time_start': 202108301600,
        # 'time_end': 202108311600,
        'time_start': 202001011600,
        'time_end': 202108311600,
        'slippage': 0,
        'mode': 1
    })


class Wt4RLSimpleEvaluator(SimpleCTASubProcessEnv):
    env_num = 1
    max_step = 16651
    if_discrete = False

    @property
    def state_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.action_space.shape[0]

    def __init__(self):# mode=3可以打开详细调试模式
        super().__init__(**{
        # 'time_start': 202108291600,
        # 'time_end': 202108301600,
        'time_start': 201901011600,
        'time_end': 202001011600,
        'slippage': 0,
        'mode': 2
    }) 


register('wt4rl-simplecta-trainer-v0', entry_point=Wt4RLSimpleTrainer)
register('wt4rl-simplecta-evaluator-v0', entry_point=Wt4RLSimpleEvaluator)


if __name__ == '__main__':
    @group()
    def run():
        pass

    @command()
    @option('--count', default=1)
    def debug(count):
        env: SimpleCTASubProcessEnv = make('wt4rl-simplecta-evaluator-v0')
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
        args.state_dim = 232
        args.action_dim = 4
        args.if_discrete = False
        args.if_per_or_gae = True
        args.target_return = 250  # inf
        # args.agent.if_use_cri_target = True
        # args.if_overwrite = False
        args.eval_times1 = 1
        args.eval_times2 = 3


        args.break_step = inf
        args.if_allow_break = True


        #
        args.gamma = 0.1 ** (1/12/8) # 8小时会跨过一次隔夜风险，既96个bar
        # args.learning_rate = 2 ** -14
        # args.gamma = 0.98 # 8小时会跨过一次隔夜风险，既96个bar
        args.learning_rate = 1e-4
        args.if_per_or_gae = True
        args.worker_num = 1 # 内存小的注意别爆内存

        args.env_num = 1
        args.target_step = args.max_step #* 2
        args.learner_gpus = (0,)
        args.workers_gpus = args.learner_gpus
        args.eval_gpu_id = 0
        
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

        train_and_evaluate(args)

    @command()
    def test():
        env = Wt4RLSimpleEvaluator(mode=3)
        agent = Agent()
        
        agent.init(net_dim=2 ** 8, state_dim=380, action_dim=10,
             learning_rate=0.1 ** (1/12/8), if_per_or_gae=True, env_num=1, gpu_id=0)
        agent.save_or_load_agent(cwd='./ppt-5/', if_save=False)
        

        # for i in range(10):  # 模拟训练10次
        #     obs = env.reset()
        #     done = False
        #     n = 0
        #     while not done:
        #         action = agent.select_action(obs)  # 模拟智能体产生动作
        #         obs, reward, done, info = env.step(action)
        #         n += 1
        #         # print('action:', action, 'obs:', obs,
        #         #     'reward:', reward, 'done:', done)
        #     print('第%s次训练完成，执行%s步, 盈亏%s。' % (i+1, n, env.assets))
        # env.close()

    run.add_command(debug)
    run.add_command(train)
    run.add_command(test)
    # run.add_command(eval)

    run()
