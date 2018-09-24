import torch
import resnext


def test_random():
    x = torch.randn(4, 3, 224, 224)
    net = resnext.ResNext(resnext.BlockA, resnext.CONFIGURATIONS[50], cardinality=4)
    net(x)


if __name__ == '__main__':
    test_random()
