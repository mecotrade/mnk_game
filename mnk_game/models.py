import torch


class Block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(Block, self).__init__()
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.conv1(out)
        out = torch.relu(out)

        out = self.dropout2(out)
        out = self.conv2(out)
        out += x
        out = torch.relu(out)

        return out


class VModel(torch.nn.Module):

    def __init__(self, input_shape, hidden_channels=64, num_blocks=4, dropout_rate=0.5):
        assert len(input_shape) == 3, f'input_shape is expected to be 3d array, found {input_shape.shape}'
        super(VModel, self).__init__()
        in_channels = input_shape[-1]
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same', bias=False)

        blocks = [Block(hidden_channels, hidden_channels, dropout_rate) for _ in range(num_blocks)]
        self.blocks = torch.nn.Sequential(*blocks)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear_v = torch.nn.Linear(in_features=hidden_channels * input_shape[0] * input_shape[1], out_features=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)

        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        v = self.linear_v(x)

        return v


class QModel(torch.nn.Module):

    def __init__(self, input_shape, num_actions, hidden_channels=64, num_blocks=4, dropout_rate=0.5):
        assert len(input_shape) == 3, f'input_shape is expected to be 3d array, found {input_shape.shape}'
        super(QModel, self).__init__()
        in_channels = input_shape[-1]
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same', bias=False)

        blocks = [Block(hidden_channels, hidden_channels, dropout_rate) for _ in range(num_blocks)]
        self.blocks = torch.nn.Sequential(*blocks)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear_q = torch.nn.Linear(in_features=hidden_channels * input_shape[0] * input_shape[1], out_features=num_actions)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)

        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        q = self.linear_q(x)
        return q


class PredictorModel(torch.nn.Module):

    def __init__(self, input_shape, num_actions, hidden_channels=64, num_blocks=4):
        assert len(input_shape) == 3, f'input_shape is expected to be 3d array, found {input_shape.shape}'
        super(PredictorModel, self).__init__()
        in_channels = input_shape[-1]
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same', bias=False)

        blocks = [Block(hidden_channels, hidden_channels) for _ in range(num_blocks)]
        self.blocks = torch.nn.Sequential(*blocks)
        self.linear_pi = torch.nn.Linear(in_features=hidden_channels * input_shape[0] * input_shape[1], out_features=num_actions)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)

        x = self.blocks(x)
        x = torch.flatten(x, 1)
        logits = self.linear_pi(x)

        return logits


class AlphaModel(torch.nn.Module):

    def __init__(self, input_shape, num_actions, hidden_channels=64, num_blocks=4):
        assert len(input_shape) == 3, f'input_shape is expected to be 3d array, found {input_shape.shape}'
        super(AlphaModel, self).__init__()
        in_channels = input_shape[-1]
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same', bias=False)

        blocks = [Block(hidden_channels, hidden_channels) for _ in range(num_blocks)]
        self.blocks = torch.nn.Sequential(*blocks)
        self.linear_pi = torch.nn.Linear(in_features=hidden_channels * input_shape[0] * input_shape[1], out_features=num_actions)
        self.linear_v = torch.nn.Linear(in_features=hidden_channels * input_shape[0] * input_shape[1], out_features=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)

        x = self.blocks(x)
        x = torch.flatten(x, 1)
        logits = self.linear_pi(x)
        v = torch.tanh(self.linear_v(x))

        return logits, v
