class Kanban():
    _securities_ = {}
    def __init__(self, code:str, preiod:str, count:int, roll:int) -> None:
        self.addSecurity(code=code, preiod=preiod, count=count)

    def addSecurity(self, code:str, preiod:str, count:int):
        security:list = self._securities_.get(code, [])
        security.append((preiod, count))

class Indicator(Kanban):
    pass