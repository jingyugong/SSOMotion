import torch
from data_loaders.tensors import collate

def pack_model_kwargs(
        joint_hints,
        xy_guidance_hints,
        guidance_hints,
        poset_hints,
        direction_hints,
        scene_hints,
        action_text,
        action,
        n_frames
    ):
    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}]
    collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                    arg, one_action, one_action_text in zip(collate_args, action, action_text)]
    joint_hints = joint_hints.reshape(n_frames, -1)
    xy_guidance_hints = xy_guidance_hints.reshape(n_frames, -1)
    guidance_hints = guidance_hints.reshape(n_frames, -1)
    direction_hints = direction_hints.reshape(-1)
    joint_hints = [joint_hints]
    xy_guidance_hints = [xy_guidance_hints]
    guidance_hints = [guidance_hints]
    poset_hints = [poset_hints]
    direction_hints = [direction_hints]
    scene_hints = [scene_hints]
    collate_args = [dict(arg, joint_hint=joint_hint, xy_guidance_hint=xy_guidance_hint, guidance_hint=guidance_hint, poset_hint=poset_hint, direction_hint=direction_hint, scene_hint=scene_hint) for arg, joint_hint, xy_guidance_hint, guidance_hint, poset_hint, direction_hint, scene_hint in zip(collate_args, joint_hints, xy_guidance_hints, guidance_hints, poset_hints, direction_hints, scene_hints)]
    _, model_kwargs = collate(collate_args)

    return model_kwargs
