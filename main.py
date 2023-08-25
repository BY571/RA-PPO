import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


@hydra.main(version_base=None, config_path="config", config_name="conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(config=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
