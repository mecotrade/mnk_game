class Context:

    def __init__(self, board):
        self.board = board
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
        return type(self)(board)
