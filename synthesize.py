import logging
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_audio")
def main(config):
    set_random_seed(config.inferencer.seed)

    device = (
        "cuda" if config.inferencer.device == "auto" and torch.cuda.is_available() else config.inferencer.device
    )

    dataloaders, batch_transforms = get_dataloaders(config, device)

    generator = instantiate(config.generator, _convert_="partial").to(device)

    metric_section = config.metrics.get("inference") or config.metrics.get("test", [])
    metrics = {"inference": [instantiate(metric) for metric in metric_section]}

    writer = None
    if "writer" in config:
        project_config = OmegaConf.to_container(config)
        logger = logging.getLogger("inference")
        logger.setLevel(logging.INFO)
        writer = instantiate(config.writer, logger, project_config)

    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        generator=generator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
        writer=writer,
    )

    for part, values in inferencer.run_inference().items():
        for key, value in values.items():
            print(f"{part}_{key}: {value}")


if __name__ == "__main__":
    main()