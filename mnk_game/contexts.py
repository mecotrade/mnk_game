class Context:

    def __init__(self, board, history: list):
        self.board = board
        self.history = history
        self.reward, self.done, self.move, self.actions = self.analyze()

    @staticmethod
    def new():
        raise NotImplementedError

    @staticmethod
    def num_actions():
        raise NotImplementedError

    def analyze(self) -> (float, bool, int, list):
        raise NotImplementedError

    def apply(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def __call__(self, action):
        board = self.apply(action)
        return type(self)(board, self.history + [action])


class ContextTree(Context):

    def __init__(self, board, history: list):
        super().__init__(board, history)
        self.parent: ContextTree | None = None
        self.value = 0
        self.visits = 0
        self.children: list = [None] * self.num_actions()

    def __call__(self, action):
        child = self.children[action]
        if child is None:
            child = super().__call__(action)
            child.parent = self
            self.children[action] = child
        return child

    def of(self, action):
        return super().__call__(action)
