import torch
from tools.project_config_tools import action_enumerator
from data_loaders.a2m.mixedmotion import MixedMotion
from data_loaders.scene2motion.humanise import HumaniseMotion

class MixedSceneMotion(torch.utils.data.Dataset):
    def __init__(
        self,
        split="train",
        num_frames=30,
        controlnet=True,
        **kargs
        ):
        self.num_actions = len(action_enumerator)
        self.datasets = []
        self.datasets.append(HumaniseMotion(split=split, num_frames=num_frames, controlnet=controlnet, **kargs))
        self.datasets.append(MixedMotion(split=split, num_frames=num_frames, controlnet=controlnet, **kargs))
        self.dataset_repeats = [7, 1]
        self.len = 0
        self.dataset_offsets = []
        for dataset, dataset_repeat in zip(self.datasets, self.dataset_repeats):
            self.len += len(dataset) * dataset_repeat
            self.dataset_offsets.append(self.len)
        return

    def __getitem__(self, index):
        for dataset_i, offset in enumerate(self.dataset_offsets):
            if index < offset:
                if dataset_i == 0:
                    ret = self.datasets[dataset_i][index % len(self.datasets[dataset_i])]
                else:
                    ret = self.datasets[dataset_i][(index - self.dataset_offsets[dataset_i - 1]) % len(self.datasets[dataset_i])]
        return ret

    def __len__(self):
        return self.len
