import sys
import fiddle as fdl

from cfg import registry


def main():
    cfg_name = sys.argv[1] if len(sys.argv) == 2 else "gpt2-117m"
    cfg = registry.get_cfg(cfg_name)
    trainer = fdl.build(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
