# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import os
import numpy as np
import torch
import torch.nn as nn
import clip
import warnings
from model.rotation2xyz import Rotation2xyz
from tools.project_config_tools import project_dir, host_device, local_grid_info, global_category2label, ablation_semantic
from .transformer import *
if ablation_semantic:
    warnings.warn("ablation semantic", UserWarning)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class JointHintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.ModuleList([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        ])

    def forward(self, x, motion_feat):
        x = x.permute((1, 0, 2))

        for module in self.poseEmbedding:
            x = module(x)  # [seqlen, bs, d]
        return x

class DirectionHintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.goalEmbedding = nn.ModuleList([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        ])

        self.num_tokens = 8
        self.cross_attention = nn.MultiheadAttention(self.latent_dim, num_heads=4, dropout=0.1, kdim=self.latent_dim//self.num_tokens, vdim=self.latent_dim//self.num_tokens)
        return

    def forward(self, x, motion_feat):
        for module in self.goalEmbedding:
            x = module(x)  # [seqlen, bs, d]

        x = x.view(-1, self.num_tokens, self.latent_dim//self.num_tokens)
        x = torch.permute(x, (1, 0, 2))
        x, _ = self.cross_attention(motion_feat, x, x)

        return x

class SharedEncoder(nn.Module):
    def __init__(self, init_dim, final_dim):
        super().__init__()
        self.init_dim = init_dim
        self.final_dim = final_dim
        self.shared_sem_encoder = nn.ModuleList([
            nn.Conv2d(self.init_dim, self.init_dim*2, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ])
        self.res_block0 = nn.ModuleList([
            nn.Conv2d(self.init_dim*2, self.init_dim*2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(self.init_dim*2, self.init_dim*2, kernel_size=3, stride=1, padding=1),
        ])
        self.conv_block0 = nn.ModuleList([
            nn.Conv2d(self.init_dim*2, self.init_dim*4, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(self.init_dim*4, self.init_dim*8, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(self.init_dim*8, self.final_dim, kernel_size=5, stride=2, padding=2),
        ])
        return
    def forward(self, x):
        for module in self.shared_sem_encoder:
            x = module(x)
        res = x
        for module in self.res_block0:
            x = module(x)
        x += res
        for module in self.conv_block0:
            x = module(x)
        x = torch.mean(x, dim=[2, 3])
        return x

class SemanticHintBlock(nn.Module):
    def __init__(self, latent_dim, dim_red=True):
        super().__init__()
        self._prepare_clip_embeddings()
        self.init_dim = 16
        self.intermediate_dim = self.init_dim * 16
        self.dim_red = dim_red
        if dim_red:
            self.lower_clip_dim = nn.Linear(512, self.init_dim)
        else:
            self.init_dim = 512
        self.shared_sem_encoder = SharedEncoder(self.init_dim, self.intermediate_dim)
        self.x_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//8))
        self.y_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//4))
        self.z_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//4))
        return

    def forward(self, x):
        if self.dim_red:
            lower_dim_semantic_features = self.lower_clip_dim(self.semantic_features)
        else:
            lower_dim_semantic_features = self.semantic_features
        b, _, h, w = x['x_map1_sem'].shape
        x_map1_feat = torch.index_select(lower_dim_semantic_features, 0, x['x_map1_sem'].view(-1)).view(b, h, w, self.init_dim).permute(0, 3, 1, 2)
        x_map1_feat = self.shared_sem_encoder(x_map1_feat)
        x_map1_feat = self.x_map_mapping(x_map1_feat)

        b, _, h, w = x['x_map2_sem'].shape
        x_map2_feat = torch.index_select(lower_dim_semantic_features, 0, x['x_map2_sem'].view(-1)).view(b, h, w, self.init_dim).permute(0, 3, 1, 2)
        x_map2_feat = self.shared_sem_encoder(x_map2_feat)
        x_map2_feat = self.x_map_mapping(x_map2_feat)

        b, _, h, w = x['y_map1_sem'].shape
        y_map1_feat = torch.index_select(lower_dim_semantic_features, 0, x['y_map1_sem'].view(-1)).view(b, h, w, self.init_dim).permute(0, 3, 1, 2)
        y_map1_feat = self.shared_sem_encoder(y_map1_feat)
        y_map1_feat = self.y_map_mapping(y_map1_feat)

        b, _, h, w = x['y_map2_sem'].shape
        y_map2_feat = torch.index_select(lower_dim_semantic_features, 0, x['y_map2_sem'].view(-1)).view(b, h, w, self.init_dim).permute(0, 3, 1, 2)
        y_map2_feat = self.shared_sem_encoder(y_map2_feat)
        y_map2_feat = self.y_map_mapping(y_map2_feat)

        b, _, h, w = x['z_map1_sem'].shape
        z_map1_feat = torch.index_select(lower_dim_semantic_features, 0, x['z_map1_sem'].view(-1)).view(b, h, w, self.init_dim).permute(0, 3, 1, 2)
        z_map1_feat = self.shared_sem_encoder(z_map1_feat)
        z_map1_feat = self.z_map_mapping(z_map1_feat)

        x = torch.cat((x_map1_feat, x_map2_feat, y_map1_feat, y_map2_feat, z_map1_feat), axis=1)

        return x

    def _prepare_clip_embeddings(self):
        self.semantic_feature_file = os.path.join(project_dir, "assets/global_semantic_clip_feature.npy")
        if os.path.exists(self.semantic_feature_file):
            self.semantic_features = np.load(self.semantic_feature_file)
            if self.semantic_features.shape[0] != len(global_category2label):
                reset_semantic_features = True
            else:
                reset_semantic_features = False
        else:
            reset_semantic_features = True

        if reset_semantic_features:
            self.semantic_features = np.zeros((len(global_category2label), 512), dtype=np.float32)
            self.clip_model, _ = clip.load("ViT-B/32", device=host_device)
            for category, category_id in global_category2label.items():
                text = clip.tokenize([category]).to(host_device)
                with torch.no_grad():
                    semantic_feature = self.clip_model.encode_text(text).cpu().numpy()
                    self.semantic_features[category_id:category_id+1] = semantic_feature
            np.save(self.semantic_feature_file, self.semantic_features)
        self.semantic_features = torch.tensor(self.semantic_features, dtype=torch.float, device=host_device, requires_grad=False)
        return

class DepthHintBlock(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.init_dim = 1
        self.intermediate_dim = self.init_dim * 16
        self.shared_depth_encoder = SharedEncoder(self.init_dim, self.intermediate_dim)
        self.x_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//8))
        self.y_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//4))
        self.z_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//4))
        return

    def forward(self, x):
        x_map1_feat = self.shared_depth_encoder(self.gaussian_kernel(x['x_map1_depth']))
        x_map1_feat = self.x_map_mapping(x_map1_feat)

        x_map2_feat = self.shared_depth_encoder(self.gaussian_kernel(x['x_map2_depth']))
        x_map2_feat = self.x_map_mapping(x_map2_feat)

        y_map1_feat = self.shared_depth_encoder(self.gaussian_kernel(x['y_map1_depth']))
        y_map1_feat = self.y_map_mapping(y_map1_feat)

        y_map2_feat = self.shared_depth_encoder(self.gaussian_kernel(x['y_map2_depth']))
        y_map2_feat = self.y_map_mapping(y_map2_feat)

        z_map1_feat = self.shared_depth_encoder(self.gaussian_kernel(x['z_map1_depth']))
        z_map1_feat = self.z_map_mapping(z_map1_feat)

        x = torch.cat((x_map1_feat, x_map2_feat, y_map1_feat, y_map2_feat, z_map1_feat), axis=1)
        return x

    def gaussian_kernel(self, x, sigma=1):
        x = torch.exp(-x**2 / (2 * sigma**2))
        return x

class ColorHintBlock(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.init_dim = 3
        self.intermediate_dim = self.init_dim * 16
        self.shared_rgb_encoder = SharedEncoder(self.init_dim, self.intermediate_dim)
        self.x_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//8))
        self.y_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//4))
        self.z_map_mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim//4))
        return
    def forward(self, x):
        x_map1_feat = self.shared_rgb_encoder(x['x_map1_rgb']/255.)
        x_map1_feat = self.x_map_mapping(x_map1_feat)

        x_map2_feat = self.shared_rgb_encoder(x['x_map2_rgb']/255.)
        x_map2_feat = self.x_map_mapping(x_map2_feat)

        y_map1_feat = self.shared_rgb_encoder(x['y_map1_rgb']/255.)
        y_map1_feat = self.y_map_mapping(y_map1_feat)

        y_map2_feat = self.shared_rgb_encoder(x['y_map2_rgb']/255.)
        y_map2_feat = self.y_map_mapping(y_map2_feat)

        z_map1_feat = self.shared_rgb_encoder(x['z_map1_rgb']/255.)
        z_map1_feat = self.z_map_mapping(z_map1_feat)

        x = torch.cat((x_map1_feat, x_map2_feat, y_map1_feat, y_map2_feat, z_map1_feat), axis=1)
        return x

class SceneHintBlock(nn.Module):
    def __init__(self, latent_dim, dim_red=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.semantic_latent_dim = self.latent_dim//2
        self.depth_latent_dim = self.latent_dim//4
        self.color_latent_dim = self.latent_dim//4
        self.semanticEmbedding = SemanticHintBlock(self.semantic_latent_dim, dim_red=dim_red)
        self.depthEmbedding = DepthHintBlock(self.depth_latent_dim)
        self.colorEmbedding = ColorHintBlock(self.color_latent_dim)

        self.num_tokens = 8
        self.cross_attention = nn.MultiheadAttention(self.latent_dim, num_heads=4, dropout=0.1, kdim=self.latent_dim//self.num_tokens, vdim=self.latent_dim//self.num_tokens)
        return

    def forward(self, x, motion_feat):
        x_semantic = self.semanticEmbedding(x)
        x_depth = self.depthEmbedding(x)
        x_color = self.colorEmbedding(x)

        x_semantic = x_semantic.view(-1, self.num_tokens, self.semantic_latent_dim//self.num_tokens)
        if ablation_semantic:
            x_semantic *= 0
        x_depth = x_depth.view(-1, self.num_tokens, self.depth_latent_dim//self.num_tokens)
        x_color = x_color.view(-1, self.num_tokens, self.color_latent_dim//self.num_tokens)

        x = torch.cat((x_semantic, x_depth, x_color), axis=-1)
        x = torch.permute(x, (1, 0, 2))
        x, _ = self.cross_attention(motion_feat, x, x)

        return x

class CMDM(torch.nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, *args, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.emb_trans_dec = emb_trans_dec

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        # --- MDM ---
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        # ------
        # --- CMDM ---
        # input 263 or 6 * 3 or 3
        n_joints = 22 if njoints in [263, 69] else 21
        self.input_joint_hint_block = JointHintBlock(self.data_rep, n_joints * 3, self.latent_dim)
        self.input_direction_hint_block = DirectionHintBlock(self.data_rep, n_joints * 3, self.latent_dim)
        self.input_scene_hint_block = SceneHintBlock(self.latent_dim)

        self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.c_seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers,
                                                    return_intermediate=True)

        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
        
        self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.c_embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            if 'action' in self.cond_mode:
                self.c_embed_action = EmbedAction(self.num_actions, self.latent_dim)

        return

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
        """
        Realism Guidance
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        emb = self.c_embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.c_embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        x = self.c_input_process(x)

        joint_seq_mask = y['joint_hint'].abs().sum(-1) != 0
        joint_guided_hint = self.input_joint_hint_block(y['joint_hint'].float(), x.clone())  # [bs, d]
        direction_mask = y['direction_hint'].abs().sum(-1) != 0
        direction_guided_hint = self.input_direction_hint_block(y['direction_hint'].float(), x.clone())
        scene_mask = y['scene_hint']['scene_valid']
        scene_guided_hint = self.input_scene_hint_block(y['scene_hint'], x.clone()+direction_guided_hint)

        all_guided_hint = joint_guided_hint * joint_seq_mask.permute(1, 0).unsqueeze(-1) + direction_guided_hint * direction_mask.unsqueeze(0).unsqueeze(-1) + scene_guided_hint * scene_mask.unsqueeze(0)
        x += all_guided_hint 

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control
    
    def mdm_forward(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, control=control)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        y['hint']: [bs, n_frames, n_joints*3]
        """
        if 'joint_hint' in y.keys() and 'direction_hint' in y.keys() and 'scene_hint' in y.keys():
            if y["joint_hint"].device != x.device:
                y["joint_hint"] = y["joint_hint"].to(x.device)
            if y["direction_hint"].device != x.device:
                y["direction_hint"] = y["direction_hint"].to(x.device)
            for key in y["scene_hint"].keys():
                if y["scene_hint"][key].device != x.device:
                    y["scene_hint"][key] = y["scene_hint"][key].to(x.device)
            control = self.cmdm_forward(x, timesteps, y)
        else:
            y_ = {}
            if 'joint_hint' not in y.keys():
                n_joints = 22 if self.njoints in [263, 69] else 21
                y_.update({'joint_hint': torch.zeros((x.shape[0], x.shape[-1], n_joints * 3), device=x.device)})
            if 'direction_hint' not in y.keys():
                y_.update({'direction_hint': torch.zeros((x.shape[0], n_joints * 3), device=x.device)})
            if 'scene_hint' not in y.keys():
                x_grids, y_grids, z_grids = local_grid_info['grid_steps']
                x_map1 = torch.zeros((x.shape[0], 5, z_grids, y_grids), device=x.device)
                x_map2 = torch.zeros((x.shape[0], 5, z_grids, y_grids), device=x.device)
                y_map1 = torch.zeros((x.shape[0], 5, z_grids, x_grids), device=x.device)
                y_map2 = torch.zeros((x.shape[0], 5, z_grids, x_grids), device=x.device)
                z_map1 = torch.zeros((x.shape[0], 5, y_grids, x_grids), device=x.device)
                void_scene_hint = {'scene_valid': torch.zeros((x.shape[0], 1), device=x.device), 'x_map1_depth': x_map1[:,0:1,:,:], 'x_map2_depth': x_map2[:,0:1,:,:], 'y_map1_depth': y_map1[:,0:1,:,:], 'y_map2_depth': y_map2[:,0:1,:,:], 'z_map1_depth': z_map1[:,0:1,:,:], 'x_map1_rgb': x_map1[:,1:4,:,:], 'x_map2_rgb': x_map2[:,1:4,:,:], 'y_map1_rgb': y_map1[:,1:4,:,:], 'y_map2_rgb': y_map2[:,1:4,:,:], 'z_map1_rgb': z_map1[:,1:4,:,:], 'x_map1_sem': x_map1[:,4:5,:,:], 'x_map2_sem': x_map2[:,4:5,:,:], 'y_map1_sem': y_map1[:,4:5,:,:], 'y_map2_sem': y_map2[:,4:5,:,:], 'z_map1_sem': z_map1[:,4:5,:,:]}
                y_.update({'scene_hint': void_scene_hint})
            y_.update(y)
            control = self.cmdm_forward(x, timesteps, y_)
        output = self.mdm_forward(x, timesteps, y, control)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec', 'mixed_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec', 'mixed_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output

class Shared3DEncoder(nn.Module):
    def __init__(self, init_dim, final_dim):
        super().__init__()
        self.init_dim = init_dim
        self.final_dim = final_dim
        self.shared_sem_encoder = nn.ModuleList([
            nn.Conv3d(self.init_dim, self.init_dim*2, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        ])
        self.res_block0 = nn.ModuleList([
            nn.Conv3d(self.init_dim*2, self.init_dim*2, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(self.init_dim*2, self.init_dim*2, kernel_size=3, stride=1, padding=1),
        ])
        self.conv_block0 = nn.ModuleList([
            nn.Conv3d(self.init_dim*2, self.init_dim*4, kernel_size=5, stride=2, padding=2),
            nn.Conv3d(self.init_dim*4, self.init_dim*8, kernel_size=5, stride=2, padding=2),
            nn.Conv3d(self.init_dim*8, self.final_dim, kernel_size=5, stride=2, padding=2),
        ])
        return
    def forward(self, x):
        for module in self.shared_sem_encoder:
            x = module(x)
        res = x
        for module in self.res_block0:
            x = module(x)
        x += res
        for module in self.conv_block0:
            x = module(x)
        x = torch.mean(x, dim=[2, 3, 4])
        return x

class SemanticHintBlockNoBTDVariant(nn.Module):
    def __init__(self, latent_dim, dim_red=True):
        super().__init__()
        self._prepare_clip_embeddings()
        self.init_dim = 16
        self.intermediate_dim = self.init_dim * 16
        self.dim_red = dim_red
        if dim_red:
            self.lower_clip_dim = nn.Linear(512, self.init_dim)
        else:
            self.init_dim = 512
        self.shared_sem_encoder = Shared3DEncoder(self.init_dim, self.intermediate_dim)
        self.mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim))
        return

    def forward(self, x):
        if self.dim_red:
            lower_dim_semantic_features = self.lower_clip_dim(self.semantic_features)
        else:
            lower_dim_semantic_features = self.semantic_features
        b, _, d, h, w = x['sem'].shape
        feat = torch.index_select(lower_dim_semantic_features, 0, x['sem'].view(-1)).view(b, d, h, w, self.init_dim).permute(0, 4, 1, 2, 3)
        feat = self.shared_sem_encoder(feat)
        feat = self.mapping(feat)

        x = feat

        return x

    def _prepare_clip_embeddings(self):
        self.semantic_feature_file = os.path.join(project_dir, "assets/global_semantic_clip_feature.npy")
        if os.path.exists(self.semantic_feature_file):
            self.semantic_features = np.load(self.semantic_feature_file)
            if self.semantic_features.shape[0] != len(global_category2label):
                reset_semantic_features = True
            else:
                reset_semantic_features = False
        else:
            reset_semantic_features = True

        if reset_semantic_features:
            self.semantic_features = np.zeros((len(global_category2label), 512), dtype=np.float32)
            self.clip_model, _ = clip.load("ViT-B/32", device=host_device)
            for category, category_id in global_category2label.items():
                text = clip.tokenize([category]).to(host_device)
                with torch.no_grad():
                    semantic_feature = self.clip_model.encode_text(text).cpu().numpy()
                    self.semantic_features[category_id:category_id+1] = semantic_feature
            np.save(self.semantic_feature_file, self.semantic_features)
        self.semantic_features = torch.tensor(self.semantic_features, dtype=torch.float, device=host_device, requires_grad=False)
        return

class ColorHintBlockNoBTDVariant(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.init_dim = 3
        self.intermediate_dim = self.init_dim * 16
        self.shared_rgb_encoder = Shared3DEncoder(self.init_dim, self.intermediate_dim)
        self.mapping = zero_module(nn.Linear(self.intermediate_dim, latent_dim))
        return
    def forward(self, x):
        feat = self.shared_rgb_encoder(x['rgb']/255.)
        feat = self.mapping(feat)

        x = feat
        return x

class SceneHintBlockNoBTDVariant(nn.Module):
    def __init__(self, latent_dim, dim_red=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.semantic_latent_dim = self.latent_dim//2
        self.color_latent_dim = self.latent_dim//2
        self.semanticEmbedding = SemanticHintBlockNoBTDVariant(self.semantic_latent_dim, dim_red=dim_red)
        self.colorEmbedding = ColorHintBlockNoBTDVariant(self.color_latent_dim)

        self.num_tokens = 8
        self.cross_attention = nn.MultiheadAttention(self.latent_dim, num_heads=4, dropout=0.1, kdim=self.latent_dim//self.num_tokens, vdim=self.latent_dim//self.num_tokens)
        return

    def forward(self, x, motion_feat):
        x_semantic = self.semanticEmbedding(x)
        x_color = self.colorEmbedding(x)

        x_semantic = x_semantic.view(-1, self.num_tokens, self.semantic_latent_dim//self.num_tokens)
        if ablation_semantic:
            x_semantic *= 0
        x_color = x_color.view(-1, self.num_tokens, self.color_latent_dim//self.num_tokens)

        x = torch.cat((x_semantic, x_color), axis=-1)
        x = torch.permute(x, (1, 0, 2))
        x, _ = self.cross_attention(motion_feat, x, x)

        return x

def prepare_ssobtd_input(x_grid_steps, y_grid_steps, z_grid_steps):
    scene_input = {
        "scene_valid": torch.zeros(batch_size, dtype=torch.float32),
        "x_map1_depth": torch.zeros((batch_size, 1, z_grid_steps, y_grid_steps), dtype=torch.float32),
        "x_map2_depth": torch.zeros((batch_size, 1, z_grid_steps, y_grid_steps), dtype=torch.float32),
        "y_map1_depth": torch.zeros((batch_size, 1, z_grid_steps, x_grid_steps), dtype=torch.float32),
        "y_map2_depth": torch.zeros((batch_size, 1, z_grid_steps, x_grid_steps), dtype=torch.float32),
        "z_map1_depth": torch.zeros((batch_size, 1, y_grid_steps, x_grid_steps), dtype=torch.float32),
        "x_map1_rgb": torch.zeros((batch_size, 3, z_grid_steps, y_grid_steps), dtype=torch.float32),
        "x_map2_rgb": torch.zeros((batch_size, 3, z_grid_steps, y_grid_steps), dtype=torch.float32),
        "y_map1_rgb": torch.zeros((batch_size, 3, z_grid_steps, x_grid_steps), dtype=torch.float32),
        "y_map2_rgb": torch.zeros((batch_size, 3, z_grid_steps, x_grid_steps), dtype=torch.float32),
        "z_map1_rgb": torch.zeros((batch_size, 3, y_grid_steps, x_grid_steps), dtype=torch.float32),
        "x_map1_sem": torch.zeros((batch_size, 1, z_grid_steps, y_grid_steps), dtype=torch.int32),
        "x_map2_sem": torch.zeros((batch_size, 1, z_grid_steps, y_grid_steps), dtype=torch.int32),
        "y_map1_sem": torch.zeros((batch_size, 1, z_grid_steps, x_grid_steps), dtype=torch.int32),
        "y_map2_sem": torch.zeros((batch_size, 1, z_grid_steps, x_grid_steps), dtype=torch.int32),
        "z_map1_sem": torch.zeros((batch_size, 1, y_grid_steps, x_grid_steps), dtype=torch.int32),
    }
    return scene_input

def prepare_sso_input(x_grid_steps, y_grid_steps, z_grid_steps):
    scene_input = {
        "scene_valid": torch.zeros(batch_size, dtype=torch.float32),
        "rgb": torch.zeros((batch_size, 3, x_grid_steps, y_grid_steps, z_grid_steps), dtype=torch.float32),
        "sem": torch.zeros((batch_size, 1, x_grid_steps, y_grid_steps, z_grid_steps), dtype=torch.int32),
    }
    return scene_input

def calc_ssobtd_dr_flops():
    scene_hint_block = SceneHintBlock(256)
    x = torch.rand(30, batch_size, 256)
    x_grid_steps, y_grid_steps, z_grid_steps = local_grid_info["grid_steps"]
    scene_input = prepare_ssobtd_input(x_grid_steps, y_grid_steps, z_grid_steps)
    scene_hint_block.eval()
    inputs = (scene_input, x)
    flops = FlopCountAnalysis(scene_hint_block, inputs)
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    return

def calc_ssobtd_flops():
    scene_hint_block = SceneHintBlock(256, dim_red=False)
    x = torch.rand(30, batch_size, 256)
    x_grid_steps, y_grid_steps, z_grid_steps = local_grid_info["grid_steps"]
    scene_input = prepare_ssobtd_input(x_grid_steps, y_grid_steps, z_grid_steps)
    scene_hint_block.eval()
    inputs = (scene_input, x)
    flops = FlopCountAnalysis(scene_hint_block, inputs)
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    return

def calc_sso_dr_flops():
    scene_hint_block = SceneHintBlockNoBTDVariant(256, dim_red=True)
    x = torch.rand(30, batch_size, 256)
    x_grid_steps, y_grid_steps, z_grid_steps = local_grid_info["grid_steps"]
    scene_input = prepare_sso_input(x_grid_steps, y_grid_steps, z_grid_steps)
    scene_hint_block.eval()
    inputs = (scene_input, x)
    flops = FlopCountAnalysis(scene_hint_block, inputs)
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    return

def calc_sso_flops():
    scene_hint_block = SceneHintBlockNoBTDVariant(256, dim_red=False)
    x = torch.rand(30, batch_size, 256)
    x_grid_steps, y_grid_steps, z_grid_steps = local_grid_info["grid_steps"]
    scene_input = prepare_sso_input(x_grid_steps, y_grid_steps, z_grid_steps)
    scene_hint_block.eval()
    inputs = (scene_input, x)
    flops = FlopCountAnalysis(scene_hint_block, inputs)
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    return

if __name__ == "__main__":
    from tools.project_config_tools import local_grid_info
    from fvcore.nn import FlopCountAnalysis
    batch_size = 1
    host_device='cpu'
    #calc_sso_flops()
    #calc_ssobtd_flops()
    calc_sso_dr_flops()
    #calc_ssobtd_dr_flops()
