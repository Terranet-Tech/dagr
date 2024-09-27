import os

import torch
import tqdm
import wandb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from pprint import pprint

from torch_geometric.data import DataLoader

from dagr.data.augment import Augmentations
from dagr.data.dsec_data import DSEC
from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA
from dagr.utils.args import FLAGS
from dagr.utils.logging import log_hparams, set_up_logging_directory
from dagr.utils.testing import run_test_with_visualization


def to_npy(detections):
    n_boxes = len(detections["boxes"])
    dtype = np.dtype(
        [
            ("t", "<u8"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("w", "<f4"),
            ("h", "<f4"),
            ("class_id", "u1"),
            ("class_confidence", "<f4"),
        ]
    )
    data = np.zeros(shape=(n_boxes,), dtype=dtype)
    data["t"] = detections["t"]
    data["x"] = detections["boxes"][:, 0]
    data["y"] = detections["boxes"][:, 1]
    data["w"] = detections["boxes"][:, 2] - data["x"]
    data["h"] = detections["boxes"][:, 3] - data["y"]
    data["class_id"] = detections["labels"]
    data["class_confidence"] = detections["scores"]
    return data


def save_detections(directory, detections):
    sequence_detections_map = dict()
    for d in tqdm.tqdm(detections, desc="compiling detections for saving..."):
        s = d["sequence"]
        if s not in sequence_detections_map:
            sequence_detections_map[s] = to_npy(d)
        else:
            sequence_detections_map[s] = np.concatenate(
                [sequence_detections_map[s], to_npy(d)]
            )

    for s, detections in sequence_detections_map.items():
        detections = detections[detections["t"].argsort()]
        np.save(directory / f"detections_{s}.npy", detections)


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
    log_hparams(args)

    print("init datasets")
    test_dataset = DSEC(root=args.dataset_directory, split="test", transform=Augmentations.transform_testing,
                        debug=False, min_bbox_diag=15, min_bbox_height=10, only_perfect_tracks=True,
                        no_eval=args.no_eval)
    test_loader = DataLoader(test_dataset, follow_batch=['bbox', "bbox0"], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    print("init net")
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()

    assert "checkpoint" in args
    checkpoint = torch.load(args.checkpoint)
    if args.use_ema:
        ema = ModelEMA(model)
        ema.ema.load_state_dict(checkpoint["ema"])
        ema.ema.cache_luts(
            radius=args.radius, height=test_dataset.height, width=test_dataset.width
        )

    detections = []
    with torch.no_grad():
        for n_us in np.linspace(0, 50000, args.num_interframe_steps):
            test_loader.dataset.set_num_us(int(n_us))
            metrics, detections_one_offset = run_test_with_visualization(
                test_loader,
                model=ema.ema if args.use_ema else model,
                dataset=args.dataset,
                name=wandb.run.name,
                compile_detections=True,
                no_eval=args.no_eval,
                log_every_n_batch=args.log_every_n_batch,
            )
            detections.extend(detections_one_offset)

            if metrics is not None:
                pprint(f"Time Window: {int(n_us)} ms \t mAP: {metrics['mAP']}")

        save_detections(output_directory, detections)