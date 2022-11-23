import h5py

import os
import re
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import glob
import string
import json
import librosa
from tqdm import tqdm
import DALI as dali_code
from utils import load, write_wav, load_lyrics, ToolFreq2Midi, mean_freq

import soundfile as sf

import logging
# logging.basicConfig(level=logging.DEBUG)


  
def getDALI(database_path, level, lang, genre):
    dali_annot_path = os.path.join(database_path, 'annot_tismir')
    dali_audio_path = os.path.join(database_path, 'audio')
    dali_data = dali_code.get_the_DALI_dataset(dali_annot_path, skip=[], keep=[])

    # get audio list
    audio_list = os.listdir(os.path.join(dali_audio_path))

    subset = list()
    duration = list()
    total_line_num = 0
    discard_line_num = 0

    for file in audio_list:
        if file.endswith('.mp3') and os.path.exists(os.path.join(dali_annot_path, file[:-4] + '.gz')):
            # get annotation for the current song
            try:
                entry = dali_data[file[:-4]]
                entry_info = entry.info

                # language filter
                if lang is not None and entry_info['metadata']['language'] != lang:
                    continue
                # genre filter
                if genre is not None and genre not in entry_info['metadata']['genres']:
                    continue

                song = {"id": file[:-4], "annot": [], "path": os.path.join(dali_audio_path, file)}
                samples = entry.annotations['annot'][level]

                notes_raw = entry.annotations['annot']["notes"]
                notes = [{"freq": note_raw['freq'][0], "pitch": ToolFreq2Midi(note_raw['freq'][0]), "time": note_raw['time']} for note_raw in
                         notes_raw]
                song["notes"] = notes

                subset.append(song)

                for sample in samples:
                    sample["duration"] = sample["time"][1] - sample["time"][0]

                    if sample["duration"] > 10.22:
                        print(sample)
                        discard_line_num += 1

                    song["annot"].append(sample)
                    duration.append(sample["duration"])

                    total_line_num += 1

                logging.debug("Successfully loaded {} songs".format(len(subset)))
            except:
                logging.warning("Error loading annotation for song {}".format(file))
                pass

    logging.debug("Scanning {} songs.".format(len(subset)))
    logging.debug("Total line num: {} Discarded line num: {}".format(total_line_num,  discard_line_num))

    return np.array(subset, dtype=object)

def maps(jsonfile, db_path):
  wordlst = list()
  notes = list()

  json_path = os.path.join(db_path + "/labels")
  song_path = os.path.join(db_path + "/songs", jsonfile[:-5] + ".wav")
                           
  y, sr = librosa.load(song_path, sr=22050, mono=True, offset=0.0, duration=None)
  timeplay = librosa.get_duration(filename=song_path)
  
  with open(os.path.join(json_path, jsonfile)) as f:
    data = json.load(f)

    song_start = data[0]['s']
    song_end = data[-1]['e']
  
    dura = song_end - song_start

    for line in data:
      for w in line['l']:  

        clean_w = re.findall("\w+", w['d'])
        if len(clean_w) != 1:
          print(f"discard {w['d']} in {jsonfile}")
          continue
        else:
          startw = (w['s'] / dura) * timeplay
          endw = (w['e'] / dura) * timeplay
          try:
            wfreq = mean_freq(y, sr, w['s'], w['e'])
            note = {"freq": wfreq, "pitch": ToolFreq2Midi(wfreq), 
                      "time": [startw, endw]}
            
            notes.append(note)
            wordlst.append({'text' : clean_w[0], 'freq' :  [wfreq, wfreq], 'time' : [startw, endw], 'index' : data.index(line), 'duration' : endw - startw})
          except:
            print(f"got error at file: {jsonfile} and word: {w['d']} ")
  
  print(f'finish {os.listdir(db_path).index(jsonfile)}')
 
  return jsonfile, wordlst, notes


from concurrent import futures

                           

                           
def extractJson(database_path):
  zaloData = list()
  jsonfiles = os.listdir(database_path + "/labels")
  dbpath = np.full((1, len(jsonfiles)), database_path, dtype=str)

  with futures.ProcessPoolExecutor() as pool:
    for path, word, notes in pool.map(maps, jsonfiles, dbpath):
      song = {'id' : path[:-5], 'annot' : word, 'path' : os.path.join(jsonfiles, path), 'notes' : notes}
      zaloData.append(song)

                           
def get_data_folds(database_path, p, extJson = ""):
    if os.path.exists(extJson):
      try:
        with open(extJson, 'rb') as f:
          dataset = np.load(f, allow_pickle=True)
      except:
        dataset = extractJson(database_path)

    else:
      dataset = extractJson(database_path)
      
    total_len = len(dataset)
    train_len = np.int(p * total_len)

    train_list = np.random.choice(dataset, train_len, replace=False)
    val_list = [elem for elem in dataset if elem not in train_list]
    logging.debug("First training song: " + str(train_list[0]["id"]) + " " + str(len(train_list[0]["annot"])) + " lines")
    logging.debug("train_list {} songs val_list {} songs".format(len(train_list), len(val_list)))
    return {"train" : train_list, "val" : val_list}


def crop(mix, targets, shapes):
    '''
    Crops target audio to the output shape required by the model given in "shapes"
    '''
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key][:, shapes["output_start_frame"]:shapes["output_end_frame"]]
    return mix, targets

def random_amplify(mix, targets, shapes, min, max):
    '''
    Data augmentation by randomly amplifying sources before adding them to form a new mixture
    :param mix: Original mixture
    :param targets: Source targets
    :param shapes: Shape dict from model
    :param min: Minimum possible amplification
    :param max: Maximum possible amplification
    :return: New data point as tuple (mix, targets)
    '''
    residual = mix  # start with original mix
    for key in targets.keys():
        if key != "mix":
            residual -= targets[key]  # subtract all instruments (output is zero if all instruments add to mix)
    mix = residual * np.random.uniform(min, max)  # also apply gain data augmentation to residual
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key] * np.random.uniform(min, max)
            mix += targets[key]  # add instrument with gain data augmentation to mix
    mix = np.clip(mix, -1.0, 1.0)
    return crop(mix, targets, shapes)


class LyricsAlignDatasets(Dataset):
    def __init__(self, dataset, partition, sr, shapes, hdf_dir, in_memory=False, dummy=False, pad_length=150):
        '''
        :param dataset:     a list of song with line level annotation
        :param sr:          sampling rate
        :param shapes:      dict, keys: "output_frames", "output_start_frame", "input_frames"
        :param hdf_dir:     hdf5 file
        :param in_memory:   load in memory or not
        :param dummy:       use a subset
        '''

        super(LyricsAlignDatasets, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        if dummy == False:
            self.hdf_file = os.path.join(hdf_dir, partition + ".hdf5")
        else:
            self.hdf_file = os.path.join(hdf_dir, partition + "_dummy.hdf5")

        self.sr = sr
        self.shapes = shapes
        self.hop = (shapes["output_frames"] // 2)
        self.in_memory = in_memory
        self.pad_length = pad_length

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_file):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_file, "w") as f:
                f.attrs["sr"] = sr

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    filename = example["id"]
                    # Load song
                    y, _ = load(example["path"], sr=self.sr, mono=True)

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["audio_name"] = example["id"]
                    grp.attrs["input_length"] = y.shape[1]

                    # word level annotation
                    annot_num = len(example["annot"])
                    lyrics = [sample["text"].encode() for sample in example["annot"]]
                    times = np.array([sample["time"] for sample in example["annot"]])

                    # note level annotation
                    notes = np.array(example["notes"])
                    note_num = len(notes)
                    pitches = np.array([note["freq"] for note in notes])
                    note_times = np.array([np.array([note['time'][0], note['time'][1]]) for note in notes])

                    grp.attrs["annot_num"] = annot_num
                    grp.attrs["note_num"] = note_num

                    # words and corresponding times
                    grp.create_dataset("lyrics", shape=(annot_num, 1), dtype='S100', data=lyrics)
                    grp.create_dataset("times", shape=(annot_num, 2), dtype=times.dtype, data=times)

                    # notes and corresponding times
                    grp.create_dataset("freqs", shape=(note_num, 1), dtype=np.short, data=pitches)
                    grp.create_dataset("note_times", shape=(note_num, 2), dtype=note_times.dtype, data=note_times)
                           
                           
                    # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate is not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_file, "r") as f:
            # length of song
            lengths = [f[str(song_idx)].attrs["input_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [( (l - self.shapes["output_frames"]) // self.hop) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

        self.shuffled_buffer = np.arange(self.length)
        self.shuffle_data_list()

    def shuffle_data_list(self):
        np.random.shuffle(self.shuffled_buffer)
    
    def vn2seq(self, text):
      seq = []
      lst_char = ['a', 'à', 'ả', 'ã', 'á', 'ạ',
      'ă', 'ằ', 'ẳ', 'ẵ', 'ắ', 'ặ',
      'â', 'ầ', 'ẩ', 'ẫ', 'ấ', 'ậ',
      'b', 'c', 'd', 'đ', 
      'e', 'è', 'ẻ', 'ẽ', 'é', 'ẹ',
      'ê', 'ề', 'ể', 'ễ', 'ế', 'ệ',
      'f', 'j', 'g', 'h', 
      'i', 'ì', 'ỉ', 'ĩ', 'í', 'ị',
      'j', 'k', 'l', 'm', 'n', 
      'o', 'ò', 'ỏ', 'õ', 'ó', 'ọ',
      'ô', 'ồ', 'ổ', 'ỗ', 'ố', 'ộ',
      'ơ', 'ờ', 'ở', 'ỡ', 'ớ', 'ợ',
      'p', 'q', 'r', 's', 't', 
      'u', 'ù', 'ủ', 'ũ', 'ú', 'ụ',
      'ư', 'ừ', 'ử', 'ữ', 'ứ', 'ự', 
      'v', 'w', 'x',
      'y', 'ỳ', 'ỷ', 'ỹ', 'ý', 'ỵ',
      'z', ' ']
      for c in text.lower():
        try:
          idx = lst_char.index(c)
        except:
          continue # remove unknown characters
        seq.append(idx)
      if len(seq) == 0:
          seq.append(27)
      return np.array(seq)

    def __getitem__(self, index):

        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        while True:
            # Loop until it finds a valid sample

            # Find out which slice of targets we want to read
            song_idx = self.start_pos.bisect_right(index)
            if song_idx > 0:
                index = index - self.start_pos[song_idx - 1]
                #print(f'index: {index}')
            # Check length of audio signal
            audio_length = self.hdf_dataset[str(song_idx)].attrs["input_length"]
            annot_num = self.hdf_dataset[str(song_idx)].attrs["annot_num"]
            target_length = self.shapes["output_frames"]

            #print(f'audio len: {audio_length} , annot num {annot_num}, target_length {target_length}')
            # Determine position where to start targets

            start_target_pos = index * self.hop
            #print(f'start target: {start_target_pos}')
            end_target_pos = start_target_pos + self.shapes["output_frames"]
            #print(f'end target: {end_target_pos}')
            # READ INPUTS
            # Check front padding
            start_pos = start_target_pos - self.shapes["output_start_frame"]

            #print(f'start_pos : {start_pos}')
            if start_pos < 0:
                # Pad manually since audio signal was too short
                pad_front = abs(start_pos)
                start_pos = 0
            else:
                pad_front = 0

            # Check back padding
            end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]

            #print(f'end_pos : {end_pos}')
            if end_pos > audio_length:
                # Pad manually since audio signal was too short
                pad_back = end_pos - audio_length
                end_pos = audio_length
            else:
                pad_back = 0

            # read audio and zero padding
            audio = self.hdf_dataset[str(song_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
            if pad_front > 0 or pad_back > 0:
                audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

            # find the lyrics within (start_target_pos, end_target_pos)
            words_start_end_pos = self.hdf_dataset[str(song_idx)]["times"][:]
            try:
                first_word_to_include = next(x for x, val in enumerate(list(words_start_end_pos[:, 0]))
                                             if val > start_target_pos/self.sr)
            except StopIteration:
  
                first_word_to_include = np.Inf

            try:        
                last_word_to_include = annot_num - 1 - next(x for x, val in enumerate(reversed(list(words_start_end_pos[:, 1])))
                                             if val < end_target_pos/self.sr)
            except StopIteration:

                last_word_to_include = -np.Inf

            targets = ""
            if first_word_to_include - 1 == last_word_to_include + 1: # the word covers the whole window
                # invalid sample, skip
                targets = None
                index = np.random.randint(self.length)
                continue
            if first_word_to_include <= last_word_to_include: # the window covers word[first:last+1]
              lyrics = self.hdf_dataset[str(song_idx)]["lyrics"][first_word_to_include:last_word_to_include+1]
              lyrics_list = [s[0].decode() for s in list(lyrics)]
              targets = " ".join(lyrics_list)
              targets = " ".join(targets.split())

            if len(targets) > 120:
                index = np.random.randint(self.length)
                continue
            seq = self.vn2seq(targets)
            break

        return audio, targets, seq
        
    def __len__(self):
          return self.length



class JamendoLyricsDataset(Dataset):
    def __init__(self, sr, shapes, hdf_dir, dataset, jamendo_dir, in_memory=False):
        super(JamendoLyricsDataset, self).__init__()
        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, dataset + ".hdf5")

        self.sr = sr
        self.shapes = shapes
        self.hop = shapes["output_frames"]
        self.in_memory = in_memory

        audio_dir = os.path.join(jamendo_dir, 'mp3')
        lyrics_dir = os.path.join(jamendo_dir, 'lyrics')
        self.audio_list = [file for file in os.listdir(audio_dir) if file.endswith('.mp3')]

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_file):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_file, "w") as f:
                f.attrs["sr"] = sr

                print("Adding audio files to dataset (preprocessing)...")
                for idx, audio_name in enumerate(tqdm(self.audio_list)):

                    # Load song
                    y, _ = load(os.path.join(audio_dir, audio_name), sr=self.sr, mono=True)

                    lyrics, words, idx_in_full = load_lyrics(os.path.join(lyrics_dir, audio_name[:-4]))
                    annot_num = len(words)

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["input_length"] = y.shape[1]
                    grp.attrs["audio_name"] = audio_name[:-4]
                    print(len(lyrics))

                    grp.create_dataset("lyrics", shape=(1, 1), dtype='S3000', data=np.array([lyrics.encode()]))
                    grp.create_dataset("idx", shape=(annot_num, 2), dtype=np.int, data=idx_in_full)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate is not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_file, "r") as f:
            # length of song
            lengths = [f[str(song_idx)].attrs["input_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [np.int(np.ceil(l / self.hop)) for l in lengths]

        self.lengths = lengths
        self.length = len(lengths)

    def __getitem__(self, index):

        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        # select song: index
        # Check length of audio signal
        audio_length = self.hdf_dataset[str(index)].attrs["input_length"]

        # number of chunks for that song
        num_chunk = self.lengths[index]

        chunks = []

        for i in np.arange(num_chunk):
            # Determine position where to start targets
            start_target_pos = i * self.hop
            end_target_pos = start_target_pos + self.shapes["output_frames"]

            # READ INPUTS
            # Check front padding
            start_pos = start_target_pos - self.shapes["output_start_frame"]
            if start_pos < 0:
                # Pad manually since audio signal was too short
                pad_front = abs(start_pos)
                start_pos = 0
            else:
                pad_front = 0

            # Check back padding
            end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
            if end_pos > audio_length:
                # Pad manually since audio signal was too short
                pad_back = end_pos - audio_length
                end_pos = audio_length
            else:
                pad_back = 0

            # read audio and zero padding
            audio = self.hdf_dataset[str(index)]["inputs"][:, start_pos:end_pos].astype(np.float32)
            audio_name = self.hdf_dataset[str(index)].attrs["audio_name"]
            lyrics = self.hdf_dataset[str(index)]["lyrics"][0, 0].decode()
            align_idx = self.hdf_dataset[str(index)]["idx"]
            if pad_front > 0 or pad_back > 0:
                audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

            chunks.append(audio)

        return chunks, align_idx, (lyrics, audio_name, audio_length)

    def __len__(self):
        return self.length
