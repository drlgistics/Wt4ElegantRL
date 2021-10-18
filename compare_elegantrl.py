from click import command, group, option
from elegantrl.agent import AgentPPO as Agent
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp
from envs_simple_cta import SimpleTrainer, SimpleEvaluator, WtDebugger
from gym import make, register


class Wt4RLSimpleTrainer(SimpleTrainer):
    env_num = 1
    max_step = 9540
    if_discrete = False

    @property
    def state_dim(self):
        return self.observation_space.shape[0]


class Wt4RLSimpleEvaluator(Wt4RLSimpleTrainer):
    max_step = 15195


register('wt4rl-simplecta-trainer-v0', entry_point=Wt4RLSimpleTrainer)
register('wt4rl-simplecta-evaluator-v0', entry_point=Wt4RLSimpleEvaluator)


if __name__ == '__main__':
    @group()
    def run():
        pass

    @command()
    @option('--count', default=1)
    def debug(count):
        env: SimpleTrainer = make('wt4rl-simplecta-trainer-v0')
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
        arguments = Arguments(
            env='wt4rl-simplecta-trainer-v0',
            # env='wt4rl-simplecta-evaluator-v0',
            agent=Agent()
        )
        arguments.eval_env = 'wt4rl-simplecta-evaluator-v0'
        # arguments.eval_env = 'wt4rl-simplecta-trainer-v0'
        arguments.env_num = 1
        arguments.max_step = 9540
        arguments.target_step = arguments.max_step * 1
        arguments.state_dim = 460
        arguments.action_dim = 10
        arguments.if_discrete = False
        arguments.target_return = 100  # inf
        arguments.learner_gpus = (0,)
        arguments.workers_gpus = arguments.learner_gpus

        arguments.gamma = 0.99
        arguments.net_dim = 2**9
        # arguments.net_dim = 2 ** 8
        # arguments.max_memo = 2 ** 22
        arguments.break_step = arguments.max_step*1000
        arguments.batch_size = 2 ** 11  # arguments.net_dim * 2
        # arguments.repeat_times = 1.5
        arguments.learning_rate = 2 ** -15

        arguments.eval_gpu_id = -1
        # arguments.eval_gap = 2 ** 9
        # arguments.eval_times1 = 2 ** 2
        # arguments.eval_times2 = 2 ** 5

        # arguments.worker_num = 4
        # arguments.target_step = arguments.env.max_step * 1
        # train_and_evaluate(arguments)

        train_and_evaluate_mp(arguments)

    run.add_command(debug)
    run.add_command(train)
    # run.add_command(eval)

    run()
