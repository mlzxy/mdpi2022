import math
import torch
from torch import nn
from qsparse import prune, quantize


def create_p_q(train_mode, epoch_size):
    def bypass(*args):
        if len(args) == 0:
            return nn.Identity()
        else:
            return args[0]

    quantize_first = train_mode.startswith("quantize")
    if "bs" in train_mode:
        bs = int(train_mode[train_mode.find("bs") + 2 :])
    else:
        bs = 64

    def q(*args, c=0):
        if "quantize" in train_mode:
            return (
                quantize(
                    timeout=epoch_size * (150 if quantize_first else 170),
                    channelwise=-1,
                    buffer_size=bs,
                )
                if len(args) == 0
                else quantize(
                    args[0],
                    timeout=epoch_size * (140 if quantize_first else 160),
                    buffer_size=1,
                    channelwise=c or 1,
                )
            )
        else:
            return bypass(*args)

    def p(*args):
        if "prune" in train_mode:
            kw = {
                "start": epoch_size * (155 if quantize_first else 140),
                "interval": epoch_size * 5,
                "repetition": 4,
                "sparsity": 0.5,
            }
            if "weight" in train_mode:
                return (
                    nn.Identity() if len(args) == 0 else prune(args[0], **kw, buffer_size=1)
                )
            elif "feat" in train_mode:
                return prune(**kw, buffer_size=bs, strict=False) if len(args) == 0 else args[0]
            elif "both" in train_mode:
                return prune(**kw, buffer_size=bs, strict=False) if len(args) == 0 else prune(args[0], **kw, buffer_size=1)
        else:
            return bypass(*args)

    return p, q


class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, train_mode='float', epoch_size=-1):
        super(ESPCN, self).__init__()
        p, q = create_p_q(train_mode, epoch_size)
        self.qin = q()
        self.first_part = nn.Sequential(
            q(nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2)),
            nn.Tanh(),
            q(),
            q(p(nn.Conv2d(64, 32, kernel_size=3, padding=3//2))),
            p(),
            nn.Tanh(),
            q(),
        )
        self.last_part = nn.Sequential(
            q(nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2)),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()
        # self._grad_record_counter = 0
        # import os
        # self.grad_recorder =  open(f"grad.{os.environ['TYPE']}.json", 'w')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.qin(x)
        x = self.first_part(x)
        # self.capture_stats(x)
        x = self.last_part(x)
        return x

    # def capture_stats(self, x):
    #     import json
    #     if self.training:
    #         if self._grad_record_counter % 10 == 0:
    #             print(
    #                 json.dumps({"stats": get_stats(x), "stage": "forward"}),
    #                 file=self.grad_recorder,
    #                 flush=True,
    #             )
    #             x.register_hook(
    #                 lambda grad: print(
    #                     json.dumps({"stats": get_stats(grad), "stage": "backward"}),
    #                     file=self.grad_recorder,
    #                     flush=True,
    #                 )
    #             )
    #         self._grad_record_counter += 1

    # def __del__(self):
    #     if self.grad_recorder is not None:
    #         self.grad_recorder.close()


def get_stats(ts):
    qs = [0.05, 0.95]
    with torch.no_grad():
        absts = ts.abs()
        return dict(
            min=torch.min(ts).item(),
            max=torch.max(ts).item(),
            l1norm=torch.mean(absts).item(),
            mean=torch.mean(ts).item(),
            quantiles={a: torch.quantile(ts, a).item() for a in qs},
            # abs_quantiles={a: torch.quantile(absts, a).item() for a in qs},
        )