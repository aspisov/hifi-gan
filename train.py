import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

import torch.multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="src/configs", config_name="train_hifigan")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    generator = instantiate(config.generator, _convert_="partial").to(device)
    discriminator = instantiate(config.discriminator, _convert_="partial").to(device)
    logger.info(generator)

    torch.backends.cudnn.benchmark = True  # remove slow_dilated_conv2d

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    d_loss_function = instantiate(config.d_loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    g_optimizer = instantiate(config.optimizer, params=trainable_params)
    g_lr_scheduler = instantiate(config.lr_scheduler, optimizer=g_optimizer)

    discriminator_trainable_params = filter(
        lambda p: p.requires_grad, discriminator.parameters()
    )
    d_optimizer = instantiate(config.optimizer, params=discriminator_trainable_params)
    d_lr_scheduler = instantiate(config.lr_scheduler, optimizer=d_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        g_criterion=loss_function,
        d_criterion=d_loss_function,
        metrics=metrics,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_lr_scheduler=g_lr_scheduler,
        d_lr_scheduler=d_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        compile_model=config.trainer.get("compile_model", False),
        amp_dtype=config.trainer.get("amp_dtype", None),
    )

    trainer.train()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
