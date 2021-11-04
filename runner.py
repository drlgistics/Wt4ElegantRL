import click

class Runner():
    def debug(self):
        print('debug')
    def train(self):
        print('train')
    def test(self):
        print('test')


@click.group()
@click.pass_context
def entry(ctx, obj:Runner=Runner()):
    assert isinstance(obj, Runner)
    ctx.ensure_object(Runner)

@entry.command()
@click.pass_context
def debug(ctx):
    ctx.obj.debug()

@entry.command()
@click.pass_context
def train(ctx):
    ctx.obj.train()

@entry.command()
@click.pass_context
def test(ctx):
    ctx.obj.test()

if __name__ == '__main__':
    entry()