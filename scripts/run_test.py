# avoid matlab error on server
import os

import torch
import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from torch_geometric.data import DataLoader

from dagr.data.augment import Augmentations
from dagr.data.dsec_data import DSEC
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.utils.args import FLAGS
from dagr.utils.logging import log_hparams, set_up_logging_directory
from dagr.utils.testing import run_test_with_visualization

if __name__ == "__main__":
    import random

    import numpy as np
    import torch_geometric

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    output_directory = set_up_logging_directory(
        args.dataset, args.task, args.output_directory, args.wnb_entity
    )

    project = f"low_latency-{args.dataset}-{args.task}"
    print(f"PROJECT: {project}")
    log_hparams(args)

    print("init datasets")
    dataset_path = args.dataset_directory.parent / args.dataset

    test_dataset = DSEC(
        args.dataset_directory,
        "test",
        Augmentations.transform_testing,
        debug=False,
        min_bbox_diag=15,
        min_bbox_height=10,
    )

    num_iters_per_epoch = 1

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()

    assert "checkpoint" in args
    checkpoint = torch.load(args.checkpoint)
    ema = ModelEMA(model)
    ema.ema.load_state_dict(checkpoint["ema"])

    if not args.skip_lut:
        ema.ema.cache_luts(
            radius=args.radius, height=test_dataset.height, width=test_dataset.width
        )

    with torch.no_grad():
        metrics = run_test_with_visualization(
            test_loader,
            model=ema.ema,
            dataset=args.dataset,
            log_every_n_batch=args.log_every_n_batch,
        )
        log_data = {f"testing/metric/{k}": v for k, v in metrics.items()}
        wandb.log(log_data)
        print(metrics['mAP'])

