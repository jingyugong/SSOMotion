from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

class VoidDataset:
    def __init__(self):
        self.dataset = None
        return

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "mixedmotion":
        from .a2m.mixedmotion import MixedMotion
        return MixedMotion
    elif name == "humanise":
        from .scene2motion.humanise import HumaniseMotion
        return HumaniseMotion
    elif name == "mixedscenemotion":
        from .scene2motion.mixed_scene_motion import MixedSceneMotion
        return MixedSceneMotion
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', controlnet=False, action_name=''):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        if controlnet:
            dataset = DATA(split=split, num_frames=num_frames, controlnet=controlnet, action_name=action_name)
        else:
            dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', controlnet=False, action_name='', num_workers=8, shuffle=True):
    dataset = get_dataset(name, num_frames, split, hml_mode, controlnet=controlnet, action_name=action_name)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True, collate_fn=collate
    )

    return loader
