from typing import Callable, Iterable, Sequence, Tuple
import pathlib
import librosa
import lmdb
import torch
import numpy as np
from audio_example import AudioExample
import os
from tqdm import tqdm
import yaml
import pickle
import pretty_midi
from absl import app, flags

from demucs import pretrained
from demucs.apply import apply_model

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path',
    None,
    help=
    'Path to a directory containing audio files - use slakh main directory to use slakh',
    required=True)

flags.DEFINE_string('output_path',
                    ".",
                    help='Output directory for the dataset',
                    required=False)

flags.DEFINE_bool('slakh',
                  False,
                  help='use slakh data processing',
                  required=False)

flags.DEFINE_string('slakh_only_tracks',
                  None,
                  help='use slakh data processing, only mix.flac tracks',
                  required=False)

flags.DEFINE_bool('source_separation',
                  False,
                  help='use source separation',
                  required=False)

flags.DEFINE_bool('normalize',
                  True,
                  help='Normalize audio files magnitude',
                  required=False)

flags.DEFINE_bool('cut_silences',
                  True,
                  help='Remove silence chunks',
                  required=False)

flags.DEFINE_integer('num_signal',
                     262144,
                     help='Number of audio samples to use during training')

flags.DEFINE_integer('sample_rate',
                     24000,
                     help='Sampling rate to use during training')

flags.DEFINE_integer(
    'ae_ratio',
    512,
    help=
    'Compression ratio of the AutoEncoder - required for processing midi files into the correct piano roll shape'
)

flags.DEFINE_integer('db_size',
                     200,
                     help='Maximum size (in GB) of the dataset')

flags.DEFINE_string(
    'emb_model_path',
    None,
    help='Embedding model path for precomputing the AE embeddings',
    required=False)

flags.DEFINE_integer('batch_size', 16, help='Number of chunks', required=False)

flags.DEFINE_multi_string(
    'ext',
    default=['wav', 'opus', 'mp3', 'aac', 'flac'],
    help='Extension to search for in the input directory')

flags.DEFINE_bool('dyndb',
                  default=False,
                  help="Allow the database to grow dynamically")

flags.DEFINE_string('device', 
                    'cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to use for embedding computation (cuda/cpu)')


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def search_for_audios(
    path_list: Sequence[str],
    extensions: Sequence[str] = [
        "wav", "opus", "mp3", "aac", "flac", "aif", "ogg"
    ],
):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f"*.{ext}"))
    audios = flatten(audios)
    audios = [str(a) for a in audios if 'MACOS' not in str(a)]
    return audios

def normalize_signal(x: np.ndarray,
                     max_gain_db: int = 30,
                     gain_margin: float = 0.9):
    peak = np.max(abs(x))
    if peak == 0:
        return x
    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return gain_margin * x * gain


def get_tracks_slakh(path):
    tracks = [os.path.join(path, subfolder) for subfolder in os.listdir(path)]
    meta = tracks[0] + "/metadata.yaml"
    ban_list = [
        "Chromatic Percussion",
        "Drums",
        "Percussive",
        "Sound Effects",
        "Sound effects",
    ]

    instr = []
    stem_list = []
    metadata = []
    total_stems = 0
    for trackfolder in tqdm(tracks):
        try:
            meta = trackfolder + "/metadata.yaml"
            with open(meta, "r") as file:
                d = yaml.safe_load(file)
            for k, stem in d["stems"].items():
                if stem["inst_class"] not in ban_list:
                    stem_list.append(trackfolder + "/stems/" + k + ".flac")
                    instr.append(stem["inst_class"])
                    metadata.append(stem)
                total_stems += 1
        except:
            print("ignoring reading folder : ", trackfolder)
            continue

    print(set(instr), "instruments remaining")
    print(total_stems, "stems in total")
    print(len(stem_list), "stems retained")

    audios = stem_list
    metadatas = [{
        "path": audio,
        "instrument": inst
    } for audio, inst in zip(audios, instr)]
    return audios, metadatas

def get_only_tracks_slakh(path, slakh_only_tracks):
    path = os.path.join(path, slakh_only_tracks)
    audios = []
    track_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    a = 0
    for folder in track_folders:
        track_path = os.path.join(path,folder,  'mix.flac')
        if os.path.exists(track_path):
            audios.append(track_path)
        a += 1
        if a == 100:
            break
    return audios


def get_midi(path, chunk_number):
    # MIDI
    split = path.split("/")
    split[-2] = "MIDI"
    length = FLAGS.num_signal / FLAGS.sample_rate
    midi_path = "/".join(split)[:-5] + ".mid"
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    tstart = chunk_number * FLAGS.num_signal / FLAGS.sample_rate
    tend = (chunk_number + 1) * FLAGS.num_signal / FLAGS.sample_rate
    out_notes = []
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = min(note.end - tstart, length)
            out_notes.append(note)
    if len(out_notes) == 0:
        return False, None
    midi_data.instruments[0].notes = out_notes
    midi_data.adjust_times([0, length], [0, length])
    return True, midi_data


def main(dummy):

    emb_model = None if FLAGS.emb_model_path is None else torch.jit.load(
        FLAGS.emb_model_path).to(FLAGS.device)

    env = lmdb.open(
        FLAGS.output_path,
        map_size=FLAGS.db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    if FLAGS.slakh == True:
        audios, metadatas = get_tracks_slakh(FLAGS.input_path)
    else:
        if FLAGS.slakh_only_tracks is None:
            audios = search_for_audios([FLAGS.input_path])
        else:
            audios = get_only_tracks_slakh(FLAGS.input_path, FLAGS.slakh_only_tracks)
        audios = map(str, audios)
        audios = map(os.path.abspath, audios)
        audios = [*audios]
        metadatas = [{"path": audio} for audio in audios]
        print(len(audios), " files found")

    if FLAGS.source_separation == True:
        demucs = pretrained.get_model('htdemucs_6s')
        demucs.to(FLAGS.device)
        demucs.eval()
        stem_chunks_buffer = [[] for _ in range(6)]

    chunks_buffer, metadatas_buffer = [], []
    midis = []
    cur_index = 0
    for i, (file, metadata) in enumerate(zip(tqdm(audios), metadatas)):

        try:
            audio = librosa.load(file, sr=FLAGS.sample_rate)[0]
        except:
            print("error loading file : ", file)
            continue

        audio = audio.squeeze()
        if audio.shape[-1] == 0:
            print("Empty file")
            continue
        if FLAGS.normalize:
            audio = normalize_signal(audio)

        # tile if audio is too short
        if audio.shape[-1] > FLAGS.num_signal:
            audio = np.pad(
                audio,
                (0, FLAGS.num_signal - audio.shape[-1] % FLAGS.num_signal))
        else:
            while audio.shape[-1] < FLAGS.num_signal:
                audio = np.concatenate([audio, audio])

        # Crop to drop last
        length = audio.shape[-1]
        if len(audio) % FLAGS.num_signal != 0:
            audio = audio[..., :FLAGS.num_signal *
                          (length // FLAGS.num_signal)]
            
        if not FLAGS.source_separation:
            chunks = audio.reshape(-1, FLAGS.num_signal)
            chunk_index = 0

            for j, chunk in enumerate(chunks):
                if FLAGS.slakh == True:
                    silence_test, midi = get_midi(file, j)
                else:
                    if FLAGS.cut_silences and np.max(abs(chunk)) < 0.05:
                        silence_test = False
                    else:
                        silence_test = True
                    midi = None

                # don't process buffer if empty slice
                if silence_test == False:
                    chunk_index += 1
                    continue

                midis.append(midi)
                chunks_buffer.append(
                    torch.from_numpy(chunk.reshape(1, FLAGS.num_signal)))
                metadatas_buffer.append(metadata)

                if len(chunks_buffer) == FLAGS.batch_size or (
                        j == len(chunks) - 1 and i == len(audios) - 1):
                    if emb_model is not None:
                        chunks_buffer_torch = (
                            torch.stack(chunks_buffer).squeeze().to(FLAGS.device))
                        z = emb_model.encode(
                            chunks_buffer_torch.reshape(-1, 1, FLAGS.num_signal))
                    else:
                        z = [None] * len(chunks_buffer)

                    for array, curz, midi, cur_metadata in zip(
                            chunks_buffer, z, midis, metadatas_buffer):

                        ae = AudioExample()
                        assert array.shape[-1] == FLAGS.num_signal
                        array = (array.cpu().numpy() * (2**15 - 1)).astype(
                            np.int16)
                        ae.put_array("waveform", array, dtype=np.int16)

                        # EMBEDDING
                        if curz is not None:
                            ae.put_array("z", curz.cpu().numpy(), dtype=np.float32)

                        # METADATA
                        cur_metadata["chunk_index"] = chunk_index
                        ae.put_metadata(cur_metadata)

                        # MIDI DATA
                        if midi is not None:

                            pr = midi.get_piano_roll(times=np.linspace(
                                0, FLAGS.num_signal /
                                FLAGS.sample_rate, FLAGS.num_signal //
                                FLAGS.ae_ratio))

                            ae.put_array("pr", pr, dtype=np.float32)

                        key = f"{cur_index:08d}"

                        with env.begin(write=True) as txn:
                            txn.put(key.encode(), bytes(ae))
                        cur_index += 1

                    chunks_buffer, midis, metadatas_buffer = [], [], []
                chunk_index += 1
        else:
            audio = torch.from_numpy(audio.reshape(1, 1, -1)).to(FLAGS.device)
            stems = apply_model(demucs, torch.cat([audio, audio], axis=1), device=FLAGS.device)

            chunks = audio.reshape(-1, FLAGS.num_signal)
            chunk_index = 0

            for j, chunk in enumerate(chunks):
                if FLAGS.cut_silences and torch.max(abs(chunk)) < 0.05:
                    silence_test = False
                else:
                    silence_test = True

                # don't process buffer if empty slice
                if silence_test == False:
                    chunk_index += 1
                    continue

                chunks_buffer.append(chunk.reshape(1, FLAGS.num_signal))
                metadatas_buffer.append(metadata)
                for i in range(6):
                    stem_chunk = stems[0, i, 0, j * FLAGS.num_signal: (j + 1) * FLAGS.num_signal]
                    stem_chunks_buffer[i].append(stem_chunk.reshape(1, FLAGS.num_signal))

                if len(chunks_buffer) == FLAGS.batch_size or (
                        j == len(chunks) - 1 and i == len(audios) - 1):
                    if emb_model is not None:
                        chunks_buffer_torch = (
                            torch.stack(chunks_buffer).squeeze().to(FLAGS.device))
                        z = emb_model.encode(
                            chunks_buffer_torch.reshape(-1, 1, FLAGS.num_signal))
                        stems_z = []
                        for i in range(6):
                            stem_chunk_buffer_torch = torch.stack(stem_chunks_buffer[i]).squeeze().to(FLAGS.device)
                            stems_z.append(emb_model.encode(stem_chunk_buffer_torch.reshape(-1, 1, FLAGS.num_signal)))
                    else:
                        z = [None] * len(chunks_buffer)
                        stems_z = []
                        for i in range(6):
                            stems_z.append([None] * len(chunks_buffer))

                    for i, array in enumerate(chunks_buffer):
                        curz = z[i]
                        cur_metadata = metadatas_buffer[i]

                        ae = AudioExample()
                        assert array.shape[-1] == FLAGS.num_signal
                        array = (array.cpu().numpy() * (2**15 - 1)).astype(
                            np.int16)
                        ae.put_array("waveform", array, dtype=np.int16)

                        # EMBEDDING
                        if curz is not None:
                            ae.put_array("z", curz.cpu().numpy(), dtype=np.float32)

                        for j in range(6):
                            stem_array = stem_chunks_buffer[j][i]
                            stem_array = (stem_array.cpu().numpy() * (2**15 - 1)).astype(np.int16)
                            ae.put_array(f"stem{j}-waveform", stem_array, dtype=np.int16)

                            stem_curz = stems_z[j][i]
                            if stem_curz is not None:
                                ae.put_array(f"stem{j}-z", stem_curz.cpu().numpy(), dtype=np.float32)

                        # METADATA
                        cur_metadata["chunk_index"] = chunk_index
                        ae.put_metadata(cur_metadata)


                        key = f"{cur_index:08d}"

                        with env.begin(write=True) as txn:
                            txn.put(key.encode(), bytes(ae))
                        cur_index += 1

                    chunks_buffer, midis, metadatas_buffer = [], [], []
                    stem_chunks_buffer = [[] for _ in range(6)]
                chunk_index += 1

    env.close()


if __name__ == '__main__':
    app.run(main)
