from torch import nn


class BlockA(nn.Module):
    def __init__(self, cardinality, chan_in, chan_out, chan_int, res_func, stride):
        super(BlockA, self).__init__()
        self.paths = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chan_in, chan_int, kernel_size=1),
            nn.BatchNorm2d(chan_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan_int, chan_int, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(chan_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan_int, chan_out, kernel_size=1),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(inplace=True),
        ) for _ in range(cardinality)])
        self.relu = nn.ReLU(inplace=True)
        self.res_func = res_func

    def forward(self, x):
        out = self.res_func(x)
        for path in self.paths:
            print(out.shape, path(x).shape)
            out = out + path(x)
        out = self.relu(out)
        return out


def _identity(x):
    return x.clone()


def _reshape(chan_in, chan_out, stride):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(chan_out)
    )


def _make_block_layer(block_factory, cardinality, chan_in, chan_out, chan_int, reduce, depth):
    first_stride = 2 if reduce else 1
    first_f = _reshape(chan_in, chan_out, first_stride) if reduce or chan_in != chan_out else _identity
    blocks = [block_factory(cardinality, chan_in, chan_out, chan_int, first_f, first_stride)] + \
             [block_factory(cardinality, chan_out, chan_out, chan_int, _identity, 1) for _ in range(depth - 1)]
    return nn.Sequential(*blocks)


class ResNext(nn.Module):
    def __init__(self, block, layer_sizes, *, n_classes=1000, cardinality=32):
        super(ResNext, self).__init__()
        self.beginning = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block_layers = nn.Sequential(
            _make_block_layer(block, cardinality, 64, 256, 4, False, layer_sizes[0]),
            _make_block_layer(block, cardinality, 256, 512, 4, True, layer_sizes[1]),
            _make_block_layer(block, cardinality, 512, 1024, 4, True, layer_sizes[2]),
            _make_block_layer(block, cardinality, 1024, 2048, 4, True, layer_sizes[3]),
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fully_connected = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.beginning(x)
        x = self.block_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


CONFIGURATIONS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
