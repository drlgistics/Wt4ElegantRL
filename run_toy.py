from runner import entry, Runner

class SimpleRunner(Runner):
    def test(self):
        print('test1')


if __name__ == '__main__':
    entry(obj=SimpleRunner())
