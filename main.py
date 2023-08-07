from dataclasses import dataclass, asdict
import sys


import fiddle as fdl
import wandb

from cfg import registry


@dataclass(frozen=True)
class WandbConfig:
    entity: str = 'misterpeddy'
    project: str = 'llm-flax-0'
    mode: str = 'online'
    notes: str = ''


def main():
    cfg_name = sys.argv[1] if len(sys.argv) == 2 else "gpt2-117m"
    cfg = registry.get_cfg(cfg_name)
    trainer = fdl.build(cfg)
    wandb.init(**asdict(WandbConfig()))
    trainer.train()


if __name__ == "__main__":
    main()
