from .base import BASEDataModule
from alm.data.vox import VoxDataset
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


flame = FLAME()


def convert_to_vertices(flame_params):
    # Rename keys to dict_keys(['pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params'])
    flame_params['pose_params'] = flame_params['pose']
    flame_params['jaw_params'] = flame_params['jaw']
    flame_params['expression_params'] = flame_params['expression']
    flame_params['eyelid_params'] = flame_params['eyelid']
    flame_params['shape_params'] =  np.zeros((1, 300)) #flame_params['shape']
    del flame_params['expression']
    del flame_params['eyelid']
    del flame_params['pose']
    del flame_params['jaw']
    del flame_params['shape']


    return flame.forward(flame_params)['vertices']

def load_data(args):
    file, root_dir, processor, templates, audio_dir, flame_dir = args
    if file.endswith('wav'):
        wav_path = os.path.join(root_dir, audio_dir, file)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        key = file.replace("wav", "npy")
        result = {}
        result["audio"] = input_values
        subject_id = "_".join(key.split("_")[:-1])
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
        
        # customized to Vox
        self.subjects = {
            'train': [
                'FaceTalk_170728_03272_TA',
                'FaceTalk_170904_00128_TA',
                'FaceTalk_170725_00137_TA',
                'FaceTalk_170915_00223_TA',
                'FaceTalk_170811_03274_TA',
                'FaceTalk_170913_03279_TA',
                'FaceTalk_170904_03276_TA',
                'FaceTalk_170912_03278_TA'
            ],
            'val': [
                'FaceTalk_170811_03275_TA',
                'FaceTalk_170908_03277_TA'
            ],
            'test': [
                'FaceTalk_170809_00138_TA',
                'FaceTalk_170731_00024_TA'
            ]
            # 'test': [
            #     'FaceTalk_170728_03272_TA',
            #     'FaceTalk_170904_00128_TA',
            #     'FaceTalk_170725_00137_TA',
            #     'FaceTalk_170915_00223_TA',
            #     'FaceTalk_170811_03274_TA',
            #     'FaceTalk_170913_03279_TA',
            #     'FaceTalk_170904_03276_TA',
            #     'FaceTalk_170912_03278_TA'
            # ]
        }

        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.template_file = 'templates.pkl'

        self.nfeats = 15069

        # load
        data = defaultdict(dict)
        with open(os.path.join(self.root_dir, self.template_file), 'rb') as fin:
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

        if True: # multi-process
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