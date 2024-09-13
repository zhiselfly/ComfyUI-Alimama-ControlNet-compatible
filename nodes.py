
import os
import torch
from safetensors import torch as sttr
import numpy as np
import comfy
import comfy.controlnet
from .model.alimama_controlnet import AlimamaControlNet
import folder_paths

def load_controlnet_mmdit(sd):
    extra_cond_channels = 0
    if 'pos_embed_input.proj.weight' in sd and 'pos_embed.proj.weight' in sd:
        extra_cond_channels = sd['pos_embed_input.proj.weight'].shape[1] - sd['pos_embed.proj.weight'].shape[1]
        extra_cond_channels = extra_cond_channels if extra_cond_channels > 0 else 0
    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
    model_config, operations, load_device, unet_dtype, manual_cast_dtype = comfy.controlnet.controlnet_config(new_sd)
    num_blocks = comfy.model_detection.count_blocks(new_sd, 'joint_blocks.{}.')
    for k in sd:
        new_sd[k] = sd[k]

    control_model = AlimamaControlNet(num_blocks=num_blocks, operations=operations, device=load_device, dtype=unet_dtype, extra_cond_channels=extra_cond_channels, **model_config.unet_config)
    control_model = comfy.controlnet.controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.SD3()
    latent_format.shift_factor = 0 #SD3 controlnet weirdness
    control = comfy.controlnet.ControlNet(control_model, compression_ratio=1, latent_format=latent_format, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control

class SD3AlimamaInpaintControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "AlimamaInpaintControlNetCompatible/loader"

    def load_controlnet(self, control_net_name):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet_data = comfy.utils.load_torch_file(controlnet_path, safe_load=True)
        if 'controlnet_cond_embedding.conv_in.weight' in controlnet_data or \
           'double_blocks.0.img_attn.norm.key_norm.scale' in controlnet_data or \
           'controlnet_blocks.0.weight' not in controlnet_data:
            raise Exception('Not a Alimama SD3 Inpaint ControlNet')
        controlnet = load_controlnet_mmdit(controlnet_data)
        return (controlnet,)



class SD3AlimamaInpaintControlNetApplyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "vae": ("VAE", ),
                             "image": ("IMAGE", ),
                             "mask": ("MASK", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "AlimamaInpaintControlNetCompatible"

    def apply_controlnet(self, positive, negative, control_net, image, mask, strength, start_percent, end_percent, vae):
        if strength == 0:
            return (positive, negative)

        if not isinstance(control_net.control_model, AlimamaControlNet):
            raise Exception('Please use SD3AlimamaInpaintControlNetLoader to load the ControlNet')

        image = image.clone().movedim(-1, 1)
        mask4img = mask[:, None, :, :]
        mask4img = torch.nn.functional.interpolate(
            mask4img, size=(image.shape[-1], image.shape[-2])
        )
        mask4img = mask4img.repeat(1, image.shape[1], 1, 1)

        image[mask4img > 0.5] = 0
        control_hint = vae.encode(image.movedim(1, -1))

        mask = 1. - torch.nn.functional.interpolate(
            mask[:, None, :, :], size=(control_hint.shape[-1], control_hint.shape[-2])
        )
        control_hint = torch.cat([control_hint, mask], dim=1)

        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), None)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


NODE_CLASS_MAPPINGS = {
    "SD3AlimamaInpaintControlNetLoader": SD3AlimamaInpaintControlNetLoader,
    "SD3AlimamaInpaintControlNetApplyAdvanced": SD3AlimamaInpaintControlNetApplyAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD3AlimamaInpaintControlNetLoader": "Alimama SD3 Inpaint ControlNet Loader",
    "SD3AlimamaInpaintControlNetApplyAdvanced": "Alimama SD3 Inpaint ControlNet Apply Advanced",
}
