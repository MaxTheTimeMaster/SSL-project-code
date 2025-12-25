import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, logdir: str):
        Path(logdir).mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(logdir)
        self.f = open(str(Path(logdir) / "metrics.jsonl"), "a", encoding="utf-8")

    def log(self, step: int, scalars: dict):
        for k, v in scalars.items():
            self.tb.add_scalar(k, float(v), step)
        self.f.write(json.dumps({"step": step, **scalars}, ensure_ascii=False) + "\n")
        self.f.flush()
        self.tb.flush()

    def close(self):
        self.tb.close()
        self.f.close()
