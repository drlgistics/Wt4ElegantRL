from click import command, Group
from elegantrl.agent import AgentPPO
from elegantrl.run import Arguments, train_and_evaluate
from envs_simple_cta import SimpleTrainer, SimpleEvaluator


__TRAINER_START__ = 202105311600
__TRAINER_END__ = 2021108311600

__EVALUATIR_STATR_ = 202103311600
__EVALUATIR_END_ = 202105311600

__BACKTEST_STATR_ = 2021108311600
__BACKTEST_END_ = 202110131600

env_instance = SimpleTrainer(
    time_start=__TRAINER_START__,
    time_end=__TRAINER_END__,
    )
env_instance.env_name='SimpleTrainer'

args = Arguments(if_on_policy=True)
args.agent = AgentPPO()
args.env = env_instance
args.agent.if_use_cri_target = True

args.learning_rate = 2 ** -15
args.batch_size = 2 ** 11
args.gamma = 0.99
args.break_step = 10000*10000+100
args.net_dimension = 2**9
train_and_evaluate(args)