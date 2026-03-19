import gin

gin.add_config_file_search_path('./diffusion/configs')

import torch
import os
import numpy as np

from diffusion.model import EDM_ADV, EDM_ADV_SS
from diffusion.utils.general import DummyAccelerator
from dataset import SimpleDataset

import argparse

from demucs import pretrained
from demucs.apply import apply_model

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="test")
parser.add_argument("--bsize", type=int, default=64)
parser.add_argument("--restart", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument('--config', action="append", default=[])

parser.add_argument("--dataset_type", type=str, default="waveform")
parser.add_argument("--db_path", type=str, default=None)
parser.add_argument("--out_path", type=str, default="./diffusion/runs")
parser.add_argument("--emb_model_path", type=str)


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def normalize(array):
    return (array - array.min()) / (array.max() - array.min())


def main(args):
    gin.parse_config_files_and_bindings(
        map(add_gin_extension, args.config),
        None,
    )

    if args.restart > 0:
        config_path = "./diffusion/runs/" + args.name + "/config.gin"
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    ######### BUILD MODEL #########
    model = EDM_ADV_SS()

    emb_model = torch.jit.load(args.emb_model_path)

    model.emb_model = emb_model

    model.accelerator = DummyAccelerator(
        device="cuda:" + str(args.gpu) if args.gpu >= 0 else
        "cpu")  #if args.use_accelerator else Accelerator()
    model = model.to(model.accelerator.device)

    ### GET AE RATIO ###
    dummy_x = torch.randn(1, 1, 4096).to(model.accelerator.device)
    z = model.emb_model.encode(dummy_x)
    ae_ratio = dummy_x.shape[-1] // z.shape[-1]

    ######### GET THE DATASET #########

    dataset = SimpleDataset(path=args.db_path, keys=["waveform", "z"])

    try:
        dataset[0]["z"]
        z_precomputed = True
    except:
        z_precomputed = False
        dataset.buffer_keys = ["waveform"]
        print(
            "Using on the fly AE encoding, training will be slow. Use split_to_lmdb.py with emb_model arg to precompute z"
        )

    dataset, valset = torch.utils.data.random_split(
        dataset,
        (len(dataset) - int(0.95 * len(dataset)), int(
            0.95 * len(dataset))))

    x_length = gin.query_parameter("%X_LENGTH")
    z_length = x_length // ae_ratio

    print("Loading Demucs model for source separation...")
    demucs = pretrained.get_model('htdemucs')
    demucs.to(model.accelerator.device)
    demucs.eval()

    def collate_fn(L):
        x = np.stack([l["waveform"] for l in L])
        x = torch.from_numpy(x).float().reshape((x.shape[0], 1, -1))

        if z_precomputed:
                z = np.stack([l["z"] for l in L])
                z = torch.from_numpy(z).float()

        batch_stems = [[] for i in range(len(demucs.sources))]
        for l in L:
            audio = torch.from_numpy(l['waveform']).float().reshape((1, 1, -1))
            audio.to(model.accelerator.device)
            with torch.no_grad():
                stems = apply_model(demucs, torch.cat([audio, audio], axis=1), device=model.accelerator.device)
            for i in range(stems.shape[1]):
                batch_stems[i].append(stems[0, i, 0, :])
        
        i0 = np.random.randint(0, x.shape[-1] // ae_ratio - z_length,
                                   x.shape[0])

        i1 = np.random.randint(0, x.shape[-1] // ae_ratio - z_length,
                                   x.shape[0])

        if z_precomputed:
            z_diff = torch.stack(
                [xc[..., i:i + z_length] for i, xc in zip(i0, z)])
        else:
            z_diff = torch.stack([
                xc[..., i * ae_ratio:i * ae_ratio + x_length]
                for i, xc in zip(i0, x)
            ])

        x_diff, x_toz = [], []

        for i in range(len(demucs.sources)):
            x_ = torch.stack(batch_stems[i])
            x_ = x_.reshape((x.shape[0], 1, -1))
            x_diff.append(torch.stack([
                xc[..., i * ae_ratio:i * ae_ratio + x_length]
                for i, xc in zip(i0, x_)
            ]))
            x_toz.append(torch.stack([
                    xc[..., i * ae_ratio:i * ae_ratio + x_length]
                    for i, xc in zip(i1, x_)
                ]))

        return {
            "x": z_diff,
            "x_time_cond": x_diff,
            "x_toz": x_toz,
        }

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.bsize,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True,
                                               collate_fn=collate_fn)

    valid_loader = torch.utils.data.DataLoader(valset,
                                               batch_size=args.bsize,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False,
                                               collate_fn=collate_fn)

    # print(next(iter(train_loader))["x"].shape)
    # print(next(iter(train_loader))["x_toz"].shape)

    ######### SAVE CONFIG #########
    model_dir = os.path.join(args.out_path, args.name)
    os.makedirs(model_dir, exist_ok=True)

    ######### PRINT NUMBER OF PARAMETERS #########
    num_el = 0
    for p in model.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    num_el = 0
    for encoder in model.encoders:
        for p in encoder.parameters():
            num_el += p.numel()
    print("Number of parameters - encoders : ", num_el / 1e6, "M")

    
    num_el = 0
    for classifier in model.classifiers:
        if classifier is not None:
            for p in classifier.parameters():
                num_el += p.numel()
    print("Number of parameters - classifiers : ", num_el / 1e6, "M")

    if model.encoders_time is not None:
        num_el = 0
        for encoder_time in model.encoders_time:
            for p in encoder_time.parameters():
                num_el += p.numel()
        print("Number of parameters - encoders_time : ", num_el / 1e6, "M")

    model.load_weights(model_dir=model_dir, restart_step=args.restart)

    ######### TRAINING #########
    d = {
        "dataset": dataset,
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": valid_loader,
        "restart_step": args.restart,
        "device": model.accelerator.device,
        "initialized": True,
    }
    model.fit_ss(**d)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
