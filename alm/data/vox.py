from .base import BASEDataModule
from alm.data.voxc import VoxDataset
from transformers import Wav2Vec2Processor
from collections import defaultdict

import os
from os.path import join as pjoin
import pickle
from tqdm import tqdm
import librosa
import numpy as np
from multiprocessing import Pool

from FLAME.FLAME import FLAME
import torch

flame_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flame = FLAME().to(flame_device)


def convert_to_vertices(flame_params):
    # Rename keys to dict_keys(['pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params'])
    new_flame_params = {}
    new_flame_params['pose_params']  = torch.zeros((flame_params['pose'].shape[0], 3), dtype=torch.float32).to(flame_device) #flame_params['pose']
    # new_flame_params['pose_params'] = torch.FloatTensor(flame_params['pose']).to(flame_device)

    new_flame_params['jaw_params'] =  torch.FloatTensor(flame_params['jaw']).to(flame_device)
    new_flame_params['expression_params'] = torch.FloatTensor(flame_params['expression']).to(flame_device)
    new_flame_params['eyelid_params'] = torch.FloatTensor(flame_params['eyelid']).to(flame_device)
    new_flame_params['cam'] = torch.FloatTensor(flame_params['cam']).to(flame_device)
    new_flame_params['shape_params'] =  torch.zeros((flame_params['shape'].shape[0], 300), dtype=torch.float32).to(flame_device)  #flame_params['shape']

    return flame.forward(new_flame_params)['vertices'].detach().cpu().numpy().reshape(-1, 15069)

# Map Vox to Vocaset
vox_vocaset_mapping = {
            'id10715': 'FaceTalk_170728_03272_TA',
            'id11181': 'FaceTalk_170904_00128_TA',
            'id11211': 'FaceTalk_170725_00137_TA',
            'id10231': 'FaceTalk_170915_00223_TA',
            'id10756': 'FaceTalk_170811_03274_TA',
            'id10931': 'FaceTalk_170913_03279_TA',
            'id10104': 'FaceTalk_170904_03276_TA',
            'id10786': 'FaceTalk_170912_03278_TA',
            'id10720': 'FaceTalk_170811_03275_TA',
            'id11182': 'FaceTalk_170809_00138_TA',

}

def load_data(args):
    file, root_dir, processor, templates, audio_dir, flame_dir = args
    if file.endswith('wav'):
        wav_path = os.path.join(root_dir, audio_dir, file)
        # Check if input_values was calculated before
        wav2vec_path = wav_path.replace("wav", "npy").replace("audio", "wav2vec")
        if os.path.exists(wav2vec_path):
            input_values = np.load(wav2vec_path)
        else:
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
            input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
            np.save(wav2vec_path, input_values)

        key = file.split("#")[0]
        result = {}
        result["audio"] = input_values
        subject_id = vox_vocaset_mapping[key]
        temp = templates[subject_id]
        result["name"] = file.replace(".wav", "")
        result["path"] = os.path.abspath(wav_path)
        result["template"] = temp.reshape((-1)) 

        flame_path = os.path.join(root_dir, flame_dir, file.replace("wav", "npz"))
    
        if not os.path.exists(flame_path):
            return None
        else:
            # Coverting FLAME to Vertice
            flame_params = np.load(flame_path)
            vertice = convert_to_vertices(flame_params)
            result["vertice"] = vertice
            return (key, result)

class VoxDataModule(BASEDataModule):
    def __init__(self,
                cfg,
                batch_size,
                num_workers,
                collate_fn = None,
                phase="train",
                **kwargs):
        super().__init__(batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = 'Vox'
        self.Dataset = VoxDataset
        self.cfg = cfg

        self.subjects = {
            'train': [
                'id10715',
                'id11181',
                'id11211',
                'id10231',
                'id10756',
                'id10931',
                'id10104',
                'id10786'
            ],
            'val': [
                'id10720',
            ],
            'test': [
                'id11182',
            ]
        }


        self.root_dir = kwargs.get('data_root', 'datasets/vox')
        self.audio_dir = 'audio'
        self.vertice_dir = 'flame'
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.template_file = 'templates.pkl'

        self.nfeats = 15069

        # load
        data = defaultdict(dict)
        with open(os.path.join('datasets/vocaset', self.template_file), 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        count = 0
        args_list = []
        for r, ds, fs in os.walk(os.path.join(self.root_dir, self.audio_dir)):
            for f in fs:
                args_list.append((f, self.root_dir, processor, templates, self.audio_dir, self.vertice_dir, ))

                # # comment off for full dataset
                # count += 1
                # if count > 10:
                #     break

        # split dataset
        self.data_splits = {
            'train':[],
            'val':[],
            'test':[],
        }

        motion_list = []

        if False: # multi-process
            with Pool(processes=os.cpu_count()) as pool:
                results = pool.map(load_data, args_list)
                for result in results:
                    if result is not None:
                        key, value = result
                        data[key] = value
        else: # single process
            for args in tqdm(args_list, desc="Loading data"):
                result = load_data(args)
                if result is not None:
                    key, value = result
                    data[key] = value
                else:
                    print(args[0])
                    print("Warning: data not found")


        # # calculate mean and std
        # motion_list = np.concatenate(motion_list, axis=0)
        # self.mean = np.mean(motion_list, axis=0)
        # self.std = np.std(motion_list, axis=0)

        for key, value in data.items():
            if key in self.subjects['train']:
                self.data_splits['train'].append(value)
            elif key in self.subjects['val']:
                self.data_splits['val'].append(value)
            elif key in self.subjects['test']:
                self.data_splits['test'].append(value)
                
        # self._sample_set = self.__getattr__("test_dataset")


    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        # question
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                # todo: config name not consistent
                self.__dict__[item_c] = self.Dataset(
                    data = self.data_splits[subset] ,
                    subjects_dict = self.subjects,
                    data_type = subset
                )
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")