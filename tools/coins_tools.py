import torch
import pytorch3d.transforms.rotation_conversions
from deps.coins.interaction.chamfer_distance import chamfer_dists
from deps.coins.interaction.transform_trainer import *

def rotation_matrix_vector_multiply(rot_mat, rot_vec):
    """Multiply rotations represented as matrix and axis-angle vector, return as axis-angle vector."""
    rotation = torch.matmul(rot_mat, pytorch3d.transforms.axis_angle_to_matrix(rot_vec))
    return pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(rotation)

def calc_interaction_loss(body, contact, object_pointclouds, scene, args, return_full=False):
    """
    Calculate interaction losses. Suppose all points in scene coordinate frame
    """
    batch_size = body.shape[0]
    contact_semantic, contact_scene = contact[:, :, 0], contact[:, :, 1]
    dists = chamfer_dists(body, object_pointclouds.reshape(batch_size, -1, 3))
    contact_dists = dists[contact_semantic > 0.5]
    loss_contact_semantic = torch.mean(torch.sqrt(contact_dists)) if len(contact_dists) else torch.tensor(0.0, device=body.device)
    # loss_penetration = calc_penetration_loss(scene_sdfs, body, thresh=0)
    sdf_values = scene.calc_sdf(body)
    loss_contact_scene = torch.tensor(0.0, device=body.device) if (contact_scene > 0.5).sum().item() < 1 else torch.mean(
        sdf_values[(contact_scene > 0.5)].abs())
    loss_penetration = torch.tensor(0.0, device=body.device) if sdf_values.lt(0.0).sum().item() < 1 else torch.mean(
        sdf_values[sdf_values < 0].abs())
    loss = loss_contact_scene * args.weight_contact_scene + loss_contact_semantic * args.weight_contact_semantic + loss_penetration * args.weight_penetration
    # print(loss_contact, loss_penetration)
    if return_full:
        return loss, loss_contact_semantic, loss_contact_scene, loss_penetration
    else:
        return loss

def posa_optimize(smplx_param, contact, pelvis_init, object_points, scene, body_model, body_mesh, args):
    """Optimize body pose, global translation, and global orientation guided by interaction-based terms"""
    batch_size = contact.shape[0]
    device = contact.device
    rotation_init = rot6d_to_mat(pelvis_init[:, :6])  # 1x3x3
    translation_init = pelvis_init[:, 6:]  # 1x3
    angle_z = torch.zeros((batch_size, 1), dtype=torch.float32, device=contact.device).requires_grad_(True)
    translation = translation_init.detach().clone().requires_grad_(True)
    body_model.reset_params(**smplx_param)
    init_body_pose = body_model.body_pose.detach().clone()
    body_model.body_pose.requires_grad_(True)

    param_list = [angle_z, translation, body_model.body_pose] if args.opt_pose else [angle_z, translation]
    optimizer = torch.optim.LBFGS(params=param_list, lr=args.lr_posa, max_iter=30) if args.optimizer == 'lbfgs' else torch.optim.Adam(params=param_list, lr=args.lr_posa)

    def closure(verbose=0):
        optimizer.zero_grad()
        rotation_z = pytorch3d.transforms.axis_angle_to_matrix(torch.cat((torch.zeros((batch_size, 2), device=device), angle_z), dim=1))
        rotation = torch.matmul(rotation_z, rotation_init)
        body_model_output = body_model(return_verts=True)
        pelvis = body_model_output.joints[:, 0, :].reshape(batch_size, 3)
        vertices_local = body_mesh.downsample(body_model_output.vertices - pelvis.unsqueeze(1))
        vertices_scene = torch.matmul(vertices_local, rotation.transpose(1, 2)) + translation.unsqueeze(1)
        loss_pose = F.mse_loss(body_model.body_pose, init_body_pose)
        loss_init = F.l1_loss(translation[:, :2], translation_init[:, :2]) + F.l1_loss(translation[:, 2], translation_init[:, 2]) * args.weight_z + angle_z.abs().mean()
        loss, loss_contact_semantic, loss_contact_scene, loss_penetration = calc_interaction_loss(vertices_scene, contact, object_points, scene, args, return_full=True)
        if args.annealing:
            # print('annealing', step / args.max_step_body)
            annealing_weight = 0 if step < args.max_step_body // 4 else ((step / args.max_step_body))
            loss_penetration = loss_penetration * annealing_weight
        loss_total = loss_contact_semantic * args.weight_contact_semantic + loss_contact_scene * args.weight_contact_scene + loss_penetration * args.weight_penetration + loss_pose * args.weight_pose + loss_init * args.weight_init
        loss_total.backward(retain_graph=True)
        if verbose:
            return loss_total, loss_contact_semantic, loss_contact_scene, loss_penetration, loss_pose, loss_init, rotation, vertices_scene
        else:
            return loss_total
    for step in range(args.max_step_body):
        optimizer.step(closure)

    loss_total, loss_contact_semantic, loss_contact_scene, loss_penetration, loss_pose, loss_init, rotation, vertices_scene = closure(verbose=1)
    if args.debug:
        print('total', loss_total.item(), 'semantic', loss_contact_semantic.item(), 'contact', loss_contact_scene.item(), 'penne', loss_penetration.item(), 'pose', loss_pose.item(), 'init', loss_init.item())
    smplx_param['body_pose'] = body_model.body_pose.detach().clone()
    smplx_param['global_orient'] = rotation_matrix_vector_multiply(rotation, smplx_param['global_orient'])
    smplx_param['transl'] = smplx_param['transl'] + translation
    # for key in smplx_param:
    #     smplx_param[key] = smplx_param[key].detach().clone()
    return loss_total.item(), smplx_param, vertices_scene.detach().clone()

def params2numpy(params):
    return {k: v.detach().cpu().numpy() if type(v)==torch.Tensor else v for k, v in params.items() }
