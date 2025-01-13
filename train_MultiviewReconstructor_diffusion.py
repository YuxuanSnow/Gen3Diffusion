import tyro
import time
import random
import os
import einops
import numpy as np

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

import torch
from core.options import AllConfigs
from core.options import Options
from core.models_timeImage_cond_xt_gof import LGM_timeimagecond_noise_gof
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from diffusers import DDPMScheduler
import logging
from accelerate.logging import get_logger
from core.utils import colormap
logger = get_logger(__name__)

from diffusers.utils import is_wandb_available
if is_wandb_available():
    os.environ["WANDB_MODE"] = "offline"
    import wandb

import kiui

def main():    
    batch_size = 1
    gradient_accumulation_steps = 1

    opt = Options(
        input_size=256,
        up_channels=(1024, 1024, 512, 256, 128),
        up_attention=(True, True, True, False, False),
        splat_size=128,
        output_size=512,
        batch_size=batch_size,
        data_mode='imagedream',
        num_views=12,
        num_epochs=200,
        workspace='mvr_pretrain_512_vae_xt_continue',
        resume='./checkpoints_obj/model_1.safetensors',
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='bf16',
        lr=1e-4,
        cam_radius=1.5,
        tracker_project_name='mvr_pretrain_512_bf16',
        lambda_lpips=1.0,
        lambda_distortion=100.0,
        lambda_normal=0.0,
        lambda_depth=0.0
    )

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        log_with='wandb'
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    model = LGM_timeimagecond_noise_gof(opt)
    model.train()

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("ashawkey/imagedream-ipmv-diffusers", subfolder="vae", revision=None)
    vae.eval()
    vae.requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)

    noise_scheduler = DDPMScheduler.from_pretrained("ashawkey/imagedream-ipmv-diffusers", subfolder="scheduler", revision=None)

    def print_model_info(model):
        print("="*20)
        print("model name: ", type(model).__name__)
        print("learnable parameters(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
        print("non-learnable parameters(M): ", sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6)
        print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
        print("model size(MB): ", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024)

    if accelerator.is_main_process:
        tracker_config = dict(vars(opt))
        accelerator.init_trackers(opt.tracker_project_name, config=tracker_config)

    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, layer initialized.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    if opt.data_mode == 'imagedream':
        from core.dataset_object_imagedream import Imagedream_LGM_relative_dataset as Dataset

    base_path = './rendering_data'
    render_path = [
        'ImagedreamLGM_Objaverse_132view', 
    ]
    
    train_dataset_list = []
    validation_dataset_list = []
    for idx, path in enumerate(render_path):
        train_dataset_list.append(Dataset(os.path.join(base_path, path), opt=opt, training=True, white_bg=True))
        validation_dataset_list.append(Dataset(os.path.join(base_path, path), opt=opt, training=False, white_bg=True))

    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    test_dataset = torch.utils.data.ConcatDataset(validation_dataset_list)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    print_model_info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 0.005
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    import math
    total_batch_size = opt.batch_size * accelerator.num_processes * opt.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / opt.gradient_accumulation_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Num processes = {accelerator.num_processes}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {opt.num_epochs}")
    logger.info(f"  Num updates steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {opt.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")

    for epoch in range(opt.num_epochs):
        
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):

            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                batch_size = data['imagedream_images_gt'].shape[0]
                num_view = data['imagedream_images_gt'].shape[1]  

                gt_image = data['imagedream_images_gt']
                gt_cameras_embedding = data["lgm_cam_poses_input_embedding"]

                input_image = data["context_image"]
                input_cameras_embedding = data['context_ray_embedding']
                
                gt_image = einops.rearrange(gt_image, "b n c h w -> (b n) c h w")
                
                gt_latents = vae.encode(gt_image).latent_dist.sample().detach() * vae.config.scaling_factor
                
                noise = torch.randn_like(gt_latents)
                timesteps = torch.randint(1, noise_scheduler.config.num_train_timesteps, (batch_size,), device=noise.device)
                timesteps = timesteps.long()
                timesteps_for_noise = einops.repeat(timesteps, 'b -> (b n)', n=(num_view))
                noisy_input = noise_scheduler.add_noise(gt_latents.to(dtype=torch.float32), noise.to(dtype=torch.float32), timesteps_for_noise).to(dtype=gt_latents.dtype)

                vae_decoded_x0 = vae.decode(1 / vae.config.scaling_factor * gt_latents).sample
                vae_decoded_xt = vae.decode(1 / vae.config.scaling_factor * noisy_input).sample
                vae_decoded_x0 = einops.rearrange(vae_decoded_x0, "(b n) c h w -> b n c h w", n=num_view)
                vae_decoded_xt = einops.rearrange(vae_decoded_xt, "(b n) c h w -> b n c h w", n=num_view)
                
                vae_decoded_x0xt = torch.cat([vae_decoded_x0, vae_decoded_xt], dim=2)
                
                context_image_duplicate = torch.cat([input_image, input_image], dim=2)
                vae_decoded_x0xt_with_clear_context = torch.cat([vae_decoded_x0xt, context_image_duplicate], dim=1)
                vae_decoded_x0xt_with_clear_context = einops.rearrange(vae_decoded_x0xt_with_clear_context, "b n c h w -> (b n) c h w")
                vae_decoded_x0xt_with_clear_context = (vae_decoded_x0xt_with_clear_context / 2 + 0.5).clamp(0, 1)
                
                import torchvision.transforms.functional as TF
                imagenet_mean =  (0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
                imagenet_std = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
                lgm_img_input_x0xt_with_context = TF.normalize(vae_decoded_x0xt_with_clear_context, imagenet_mean, imagenet_std)
                lgm_img_input_x0xt_with_context = einops.rearrange(lgm_img_input_x0xt_with_context, "(b n) c h w -> b n c h w", n=num_view+1)

                camera_embeddings_with_context = torch.cat([gt_cameras_embedding, input_cameras_embedding], dim=1)

                lgm_input_with_context = torch.cat([lgm_img_input_x0xt_with_context, camera_embeddings_with_context], dim=2)

                timestep = einops.rearrange(timesteps_for_noise, '(b n) -> b n', b=batch_size)
                timestep_with_ip = timestep.new_zeros((batch_size, 1), dtype=timestep.dtype)
                timestep_with_ip = torch.cat([timestep, timestep_with_ip], dim=1)
                timestep_with_ip = einops.rearrange(timestep_with_ip, 'b nv -> (b nv)', nv=num_view+1)

                data['input'] = lgm_input_with_context

                out = model(data, timestep_with_ip, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                loss_rgb = out['loss_rgb']
                if opt.lambda_distortion > 0:
                    loss_distortion = out['loss_distortion'] * opt.lambda_distortion
                if opt.lambda_normal > 0:
                    loss_normal_world = out['loss_normal_world'] * opt.lambda_normal
                if opt.lambda_lpips > 0:
                    loss_lpips = out['loss_lpips'] * opt.lambda_lpips
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

                logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr'],
                        "psnr": psnr.detach().item(), 
                        "loss_rgb": loss_rgb.detach().item(),
                        "loss_distortion": loss_distortion.detach().item() if opt.lambda_distortion > 0 else 0.0,
                        "loss_normal_world": loss_normal_world.detach().item() if opt.lambda_normal > 0 else 0.0,
                        "loss_lpips": loss_lpips.detach().item() if opt.lambda_lpips > 0 else 0.0,
                        "epoch": epoch}
                accelerator.log(logs)

            if accelerator.is_main_process:

                if i % 2000 == 0:

                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    logger.info(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f} psnr: {psnr.item():.4f}")
                
                    gt_images = data['images_output'].detach().cpu().numpy()
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/{epoch}_images_gt.jpg', gt_images)

                    pred_images = out['image'].detach().cpu().numpy()
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/{epoch}_images_pred.jpg', pred_images)
        
            
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        accelerator.wait_for_everyone()
        if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
            save_path = f'{opt.workspace}/{epoch}'
            accelerator.save_model(model, save_path)

if __name__ == "__main__":
    main()
