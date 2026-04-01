# Automatic Generation of Cover Songs Using Deep Learning

You can listen to examples [here](https://eyeless-r.github.io/Automatic-Generation-of-Cover-Songs/)

## Model training

Prior to training, install the required dependancies using :

```bash
pip install -r "requirements.txt"
```

Fine-tuning the model requires two steps : processing the dataset, then fine-tuning pretrained diffusion model.

### Dataset preparation

(after downloading Slakh2100  [here](http://www.slakh.com/)) :

For train_diffusion.py:
```bash
python dataset/split_to_lmdb.py --input_path slakh2100_flac_redux --output_path lmdb --emb_model_path autoencoder_checkpoints/AE_slakh.pt --slakh_only_tracks
```

For train_with_ss.py:
```bash
python dataset/split_to_lmdb.py --input_path slakh2100_flac_redux --output_path lmdb_ss --emb_model_path autoencoder_checkpoints/AE_slakh.pt --slakh_only_tracks True --source_separation True
```

### Diffusion model fine-tuning
The model training is configured with gin config files.

To fine-tune model without sourse separation:
```bash
python train_diffusion.py --name fine-tune --db_path lmdb --config diffusion/runs/fine-tune/config --emb_model_path autoencoder_checkpoints/AE_slakh.pt --dataset_type waveform --gpu 0 --restart 500000
```

To fine-tune model with sourse separation:
```bash
python train_with_ss.py --name fine-tune --db_path lmdb_ss --config diffusion/runs/fine-tune/config_ss --emb_model_path autoencoder_checkpoints/AE_slakh.pt --dataset_type waveform --gpu 0 --restart 500000
```

<!-- ## Inference and pretrained models

Three pretrained models are currently available : 
1. Audio to audio transfer model trained on [Slakh](http://www.slakh.com/)
2. Audio to audio transfer model trained on multiple datasets (Maestro, URMP, Filobass, GuitarSet...)
3. MIDI-to-audio model trained on [Slakh](http://www.slakh.com/)

You can download the autoencoder and diffusion model checkpoints [here](https://nubo.ircam.fr/index.php/s/8xaXbQtcY4n3Mg9/download). Make sure you copy the pretrained models in `./pretrained`. The notebooks in `./notebooks` demonstrate how to load a model and generate audio from midi and audio files. -->
