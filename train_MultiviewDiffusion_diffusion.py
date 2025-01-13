import argparse
from pathlib import Path
import os
from packaging import version
import torch
import math
from tqdm import tqdm
import einops
import torch.nn.functional as F
import shutil

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel
import logging
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
import diffusers
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
import itertools

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPVisionModel,
    CLIPTokenizer,
)
from mvdream.mv_unet import MultiViewUNetModel
from core.dataset_object_imagedream import Imagedream_LGM_relative_dataset
from imagedream_function import CLIP_preprocess
from mvdream.pipeline_imagedream import ImageDreamPipeline

logger = get_logger(__name__)

if is_wandb_available():
    os.environ["WANDB_MODE"] = "offline"
    import wandb

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Argparser for ImageDream (diffusers) training script.")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800"
    )
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=10, help=("Max number of checkpoints to store."))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=0.5, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--set_grads_to_none", default=True)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def _encode_text_prompt(
        tokenizer,
        text_encoder,
        prompt,
        device,
        batch_size
    ):
    assert isinstance(prompt, str)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )

    if (
        hasattr(text_encoder.config, "use_attention_mask")
        and text_encoder.config.use_attention_mask
    ):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, batch_size, 1)
    prompt_embeds = prompt_embeds.view(
        bs_embed * batch_size, seq_len, -1
    )

    return prompt_embeds

def main(args):
    args.pretrained_model_name_or_path = "ashawkey/imagedream-ipmv-diffusers"

    args.max_train_steps = None
    args.num_train_epochs = 100
    args.learning_rate = 1e-4
    args.mixed_precision = 'bf16'
    
    args.output_dir = "mvd_pretrain"
    args.tracker_project_name = "train_mvd_pretrain"
    
    args.num_gpu = 1
    args.train_batch_size = 1
    args.gradient_accumulation_steps = 1
    args.enable_xformers_memory_efficient_attention = True

    args.resolution = 256
    args.output_dir = args.output_dir +"_bs_"+str(args.train_batch_size * args.num_gpu * args.gradient_accumulation_steps)
    args.tracker_project_name = args.tracker_project_name + "_bs_" + str(args.train_batch_size * args.num_gpu * args.gradient_accumulation_steps)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token, private=True
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", revision=None)
    image_encoder = CLIPVisionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=None)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None)
    feature_extractor = None
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    unet = MultiViewUNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None)

    print("===========================================")
    print("Load pretrained human Imagedream Unet Model")

    from safetensors.torch import load_file
    ckpt_mvd_2k2k = load_file('checkpoints_obj/model.safetensors', device='cpu') # load the pretrained MVD UNet
    state_dict = unet.state_dict()
    for k, v in ckpt_mvd_2k2k.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')

    logger.info("Unet Models loaded from MVD pretraining successfully.")

    vae.eval()
    vae.requires_grad_(False)

    image_encoder.eval()
    image_encoder.requires_grad_(False)

    text_encoder.eval()
    text_encoder.requires_grad_(False)

    unet.train()
    unet.requires_grad_(True)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=MultiViewUNetModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_slicing()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        [{"params": unet.parameters(), "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    def print_model_info(model):
        print("="*20)
        print("model name: ", type(model).__name__)
        print("learnable parameters(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
        print("non-learnable parameters(M): ", sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6)
        print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
        print("model size(MB): ", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024)

    print_model_info(unet)
    print_model_info(vae)
    print_model_info(image_encoder)

    from core.options import Options
    opt = Options()

    # please specify MVRendering path here
    base_path = './rendering_data'
    render_path = [
        'ImagedreamLGM_Objaverse_132view', 
    ]

    train_dataset_list = []
    for idx, path in enumerate(render_path):
        train_dataset_list.append(Imagedream_LGM_relative_dataset(os.path.join(base_path, path), opt=opt, training=True, white_bg=True))

    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=1,
    )

    total_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    pct_start = 0.005
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=pct_start)

    unet, optimizer, train_dataloader, scheduler = accelerator.prepare(unet, optimizer, train_dataloader, scheduler)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    assert args.num_gpu == accelerator.num_processes, "Number of the GPU in args is false, name of logging dir is wrong."
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    do_classifier_free_guidance = args.guidance_scale > 1.0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num updates steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Num updates steps per epoch calculate = {math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f" do_classifier_free_guidance = {do_classifier_free_guidance}")
    logger.info(f" conditioning_dropout_prob = {args.conditioning_dropout_prob}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                batch_size = batch['imagedream_images_gt'].shape[0]
                num_view = batch['imagedream_images_gt'].shape[1]
                actual_num_frames = num_view + 1
                batch_orthogonal_size = batch_size * num_view

                gt_image = batch["imagedream_images_gt"].to(dtype=weight_dtype)
                input_image = batch["context_image"].squeeze(dim=1).to(dtype=weight_dtype)
                gt_pose = batch["imagedream_cam_poses_gt"].to(dtype=weight_dtype)
                text_prompt = "" + ", 3d asset photorealistic human scan"

                gt_image = einops.rearrange(gt_image, "b n c h w -> (b n) c h w")
                gt_pose = einops.rearrange(gt_pose, "b n x y -> (b n) x y")

                gt_latents = vae.encode(gt_image).latent_dist.sample().detach() * vae.config.scaling_factor

                noise = torch.randn_like(gt_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=gt_latents.device)
                timesteps = timesteps.long()
                timesteps = einops.repeat(timesteps, 'b -> (b n)', n=num_view)

                noisy_latents = noise_scheduler.add_noise(gt_latents.to(dtype=torch.float32), noise.to(dtype=torch.float32), timesteps).to(dtype=gt_latents.dtype)

                if do_classifier_free_guidance:
                    random_p = torch.rand(batch_size, device=gt_latents.device)
                    prompt_mask_img = random_p < 2*args.conditioning_dropout_prob
                    prompt_mask_img = prompt_mask_img.reshape(batch_size, 1, 1, 1)
                    random_color_input_image = torch.tensor([1, 1, 1], device=gt_latents.device, dtype=gt_latents.dtype)
                    random_color_input_image = random_color_input_image.reshape(1, 3, 1, 1)
                    random_color_input_image = random_color_input_image.repeat(batch_size, 1, args.resolution, args.resolution)

                    new_image_after_dropout = torch.where(prompt_mask_img, random_color_input_image, input_image)

                    image_clip = CLIP_preprocess(new_image_after_dropout)
                    image_clip_embedding = image_encoder(image_clip, output_hidden_states=True).hidden_states[-2]
                    
                    image_latents = vae.encode(new_image_after_dropout).latent_dist.sample() * vae.config.scaling_factor
                   
                    text_prompt_embeds = _encode_text_prompt(tokenizer, text_encoder, text_prompt, gt_latents.device, batch_size)

                else:
                    assert False, "Not implemented yet"

                noisy_gt_latents = einops.rearrange(noisy_latents, "(b n) c h w -> b n c h w", n=num_view)
                clear_image_latents = einops.rearrange(image_latents, "b c h w -> b 1 c h w")
                noisy_latents_with_ip_latent = torch.cat([noisy_gt_latents, clear_image_latents], dim=1)
                noisy_latents_with_ip_latent = einops.rearrange(noisy_latents_with_ip_latent, "b nv c h w -> (b nv) c h w")
                timestep = einops.rearrange(timesteps, '(b n) -> b n', b=batch_size)
                timestep_with_ip = timestep.new_zeros((batch_size, 1), dtype=timestep.dtype)
                timestep_with_ip = torch.cat([timestep, timestep_with_ip], dim=1)
                timestep_with_ip = einops.rearrange(timestep_with_ip, 'b nv -> (b nv)', nv=actual_num_frames)

                camera_pose_ = gt_pose.view(batch_size, 4, 16)
                padding = [0] * (len(camera_pose_.shape) * 2)
                padding[-3] = 1
                padding_tuple = tuple(padding)
                camera = F.pad(camera_pose_, padding_tuple).to(dtype=gt_latents.dtype, device=gt_latents.device)
                camera = einops.rearrange(camera, "b nv c -> (b nv) c")

                latent_model_input_with_clear_ip = torch.cat([noisy_latents_with_ip_latent])
                unet_inputs = {
                    'x': latent_model_input_with_clear_ip,
                    'timesteps': timestep_with_ip,
                    'context': torch.cat([text_prompt_embeds] * actual_num_frames),
                    'num_frames': actual_num_frames,
                    'camera': torch.cat([camera]),
                    'ip': torch.cat([image_clip_embedding] * actual_num_frames),
                    'ip_img': torch.cat([image_latents]),
                }

                unet_noise_pred = unet.forward(**unet_inputs)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                unet_noise_pred = einops.rearrange(unet_noise_pred, "(b nv) c h w -> b nv c h w", nv=actual_num_frames)
                unet_orthogonal_noise_pred = unet_noise_pred[:, :-1, :, :, :]
                unet_orthogonal_noise_pred = einops.rearrange(unet_orthogonal_noise_pred, "b nv c h w -> (b nv) c h w")
                loss = F.mse_loss(unet_orthogonal_noise_pred.float(), target.float(), reduction="none")
                loss = (loss.mean([1, 2, 3])).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(unet.parameters(), image_encoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 1:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    logger.info(f"[INFO] {step}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} loss: {loss.item():.6f}")

            loss_epoch += loss.detach().item()
            num_train_elems += 1

            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr'],
                    "loss_epoch": loss_epoch / num_train_elems,
                    "epoch": epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = ImageDreamPipeline(
                    vae=accelerator.unwrap_model(vae),
                    unet=accelerator.unwrap_model(unet),
                    image_encoder=accelerator.unwrap_model(image_encoder),
                    tokenizer=tokenizer,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    scheduler=noise_scheduler
                )
        pipeline_save_path = os.path.join(args.output_dir, f"pipeline-{global_step}")
        pipeline.save_pretrained(pipeline_save_path)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=pipeline_save_path,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)