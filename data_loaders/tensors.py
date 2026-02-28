import numpy as np
import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'joint_hint' in notnone_batches[0] and notnone_batches[0]['joint_hint'] is not None:
        joint_hint = [b['joint_hint']for b in notnone_batches]
        cond['y'].update({'joint_hint': torch.as_tensor(np.array(joint_hint))})

    if 'direction_hint' in notnone_batches[0] and notnone_batches[0]['direction_hint'] is not None:
        direction_hint = [b['direction_hint']for b in notnone_batches]
        cond['y'].update({'direction_hint': torch.as_tensor(np.array(direction_hint))})

    if 'xy_guidance_hint' in notnone_batches[0] and notnone_batches[0]['xy_guidance_hint'] is not None:
        xy_guidance_hint = [b['xy_guidance_hint']for b in notnone_batches]
        cond['y'].update({'xy_guidance_hint': torch.as_tensor(np.array(xy_guidance_hint))})

    if 'guidance_hint' in notnone_batches[0] and notnone_batches[0]['guidance_hint'] is not None:
        guidance_hint = [b['guidance_hint']for b in notnone_batches]
        cond['y'].update({'guidance_hint': torch.as_tensor(np.array(guidance_hint))})

    if 'poset_hint' in notnone_batches[0] and notnone_batches[0]['poset_hint'] is not None:
        poset_hint = [b['poset_hint']for b in notnone_batches]
        inpainted_motion = torch.as_tensor(np.array(poset_hint)).permute(0, 2, 1).unsqueeze(2).float()
        inpainting_mask = (inpainted_motion.abs().sum(1,keepdim=True) > 0.).repeat(1,69,1,1)
        cond['y'].update({'inpainted_motion': inpainted_motion})
        cond['y'].update({'inpainting_mask': inpainting_mask})

    if 'scene_hint' in notnone_batches[0] and notnone_batches[0]['scene_hint'] is not None:
        scene_hint = {}
        for k in notnone_batches[0]['scene_hint']:
            scene_hint[k] = [b['scene_hint'][k] for b in notnone_batches]
            if type(notnone_batches[0]['scene_hint'][k]) == torch.Tensor:
                scene_hint[k] = torch.stack(scene_hint[k])
            elif type(notnone_batches[0]['scene_hint'][k]) == np.ndarray:
                scene_hint[k] = torch.as_tensor(np.array(scene_hint[k]))
            elif type(notnone_batches[0]['scene_hint'][k]) in [int, float, np.float64]:
                scene_hint[k] = torch.as_tensor(np.array(scene_hint[k]))
            else:
                raise TypeError('unknown type {} in scene_hint'.format(type(notnone_batches[0]['scene_hint'][k])))
            if scene_hint[k].dtype == torch.float64:
                scene_hint[k] = scene_hint[k].float()
        cond['y'].update({'scene_hint': scene_hint})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


