## NF4 model loader of ComfyUI

Both Checkpoint and UNET loader are included.<br/>
Same as https://github.com/silveroxides/ComfyUI_bnb_nf4_fp4_Loaders, but fixed a RuntimeError.<br/>
```RuntimeError: All input tensors need to be on the same GPU, but found some tensors to not be on a GPU.```<br/>
Recommend FluxFusion to accelerate generating: https://huggingface.co/Anibaaal/Flux-Fusion-DS-merge-gguf-nf4-fp4-fp8-fp16<br/>
Recommend GGUF to generate more precisely (but more slowly than sft): https://github.com/city96/ComfyUI-GGUF<br/>

## Usage
1. Run `pip install bitsandbytes` when newly installed.
2. Make sure your ComfyUI is updated, launching ComfyUI.
3. Replace vanilla model loader node with the new node:
    * "CheckpointLoaderNF4": Load NF4 fusion model placed at `checkpoints` folder
    * "UNETLoaderNF4": Load NF4 unet model placed at `unet` folder, this also requires additional CLIP and VAE loaded.
        * VAE: downloaded into `vae` folder, loaded with vanilla VAE loader.
            * https://huggingface.co/StableDiffusionVN/Flux/blob/main/Vae/flux_vae.safetensors
        * CLIP: download into `clip` or `text_encoders` folder, loaded with vanilla DualCLIP loader.
            * t5xxl: https://huggingface.co/silveroxides/CLIP-Collection/blob/main/t5xxl_flan_latest-fp8_e4m3fn.safetensors
            * clip_l: https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14/blob/main/ViT-L-14-GmP-SAE-TE-only.safetensors

## Alternaitives
* [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku): A new quantizing method available with CUDA over 20 series, faster and more effective, working with specialized base model: [svdq-flux-dev](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev). Shamely there is no quantizing tools released causes only few model available.
* [HiDream-I1-nf4](https://github.com/lum3on/comfyui_HiDream-Sampler): A NF4 quantized version of a new 17B model larger than 12B Flux. Not working on Windows for a required package [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) is not under maintenance.

## Credits
Code adapted from the implementation by Illyasviel at [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).

## Tips
If you get OOM on first task with your low RAM / VRAM, increasing paging file then disabling CUDA - Sysmem Fallback Policy may be helpful.<br/>
Here is how to: https://support.cognex.com/docs/deep-learning_330/web/EN/deep-learning/Content/deep-learning-Topics/optimization/gpu-disable-shared.htm
