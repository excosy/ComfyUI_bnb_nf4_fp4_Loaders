## OUTDATED
Please look at [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) for more effective.

The only change except workflow nodes is [svdq-flux-dev](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev) instead of vanilla flux-dev model is required.

Like FluxFusion, there are also a faster version of svdq-flux-dev: [svdq-shuttle-jaguar](https://huggingface.co/mit-han-lab/svdq-int4-shuttle-jaguar)

The following is the original description.

## NF4 model loader of ComfyUI

Both Checkpoint and UNET loader are included.

Same as [ComfyUI_bnb_nf4_fp4_Loaders](https://github.com/silveroxides/ComfyUI_bnb_nf4_fp4_Loaders), but fixed a RuntimeError.

```RuntimeError: All input tensors need to be on the same GPU, but found some tensors to not be on a GPU.```

Recommend [FluxFusion](https://huggingface.co/Anibaaal/Flux-Fusion-DS-merge-gguf-nf4-fp4-fp8-fp16) to accelerate generating.

Recommend [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) to generate more precisely (but more slowly than sft).

## Usage
1. Run `pip install bitsandbytes` when newly installed.
2. Make sure your [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is updated, launching ComfyUI.
3. With workflows, replace vanilla model loader node with the new node:
    * "CheckpointLoaderNF4": Load NF4 AIO model placed at `checkpoints` folder
    * "UNETLoaderNF4": Load NF4 UNET model placed at `unet` folder, this also requires additional CLIP and VAE loaded.
        * VAE: downloaded [flux_vae](https://huggingface.co/StableDiffusionVN/Flux/blob/main/Vae/flux_vae.safetensors) into `vae` folder, loaded with vanilla VAE loader.
        * CLIP: download [t5xxl](https://huggingface.co/silveroxides/CLIP-Collection/blob/main/t5xxl_flan_latest-fp8_e4m3fn.safetensors) and [clip_l](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14/blob/main/ViT-L-14-GmP-SAE-TE-only.safetensors) into `clip` or `text_encoders` folder, loaded with vanilla DualCLIP loader.

## Credits
Code adapted from the implementation by Illyasviel at [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).

## Known Issues
If you get OOM on first task with your low RAM / VRAM, increasing paging file then disabling CUDA - Sysmem Fallback Policy may be helpful.<br/>
Here is how to: https://support.cognex.com/docs/deep-learning_330/web/EN/deep-learning/Content/deep-learning-Topics/optimization/gpu-disable-shared.htm
