import logging
import torch
from torch.distributed.pipelining import ScheduleGPipe

logger = logging.getLogger(__name__)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(10, 3)
        self.layers = torch.nn.ModuleList(
            Layer() for _ in range(2)
        )
        self.lm = LMHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm(x)
        return x

from torch.distributed.pipelining import pipeline, SplitPoint

# An example micro-batch input
x = torch.LongTensor([1, 2, 4, 5])

pipe = pipeline(
    module=mod,
    mb_args=(x,),
    split_spec={
        "layers.1": SplitPoint.BEGINNING,
    }
)

print(pipe)