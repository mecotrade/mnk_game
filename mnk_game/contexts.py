class Context:

    X_MOVE = 1
    O_MOVE = -1

    def __init__(self, board, history: list | None = None):
        self.board = board
        self.history = history or list()
        self.reward, self.done, self.move, self.actions = self.analyze()

    @classmethod
    def new(cls):
        raise NotImplementedError

    @classmethod
    def num_actions(cls):
        raise NotImplementedError

    @classmethod
    def shape(cls) -> tuple:
        raise NotImplementedError

    def features(self):
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

    def __init__(self, board, history: list | None = None):
        super().__init__(board, history)
        self.parent: ContextTree | None = None
        self.value = 0
        self.visits = 0
        self.children: list = [None] * self.num_actions()

    def __call__(self, action):
        child = self.children[action]
        if child is None:
            child = Context.__call__(self, action)
            child.parent = self
            self.children[action] = child
        return child

    def of(self, action):
        return Context.__call__(self, action)


class ContextPredictor(ContextTree):

    def __init__(self, board, history: list | None = None):
        super().__init__(board, history)
        self.predictor = self.uniform_predictor()

    @classmethod
    def uniform_predictor(cls):
        return [1 / cls.num_actions()] * cls.num_actions()
