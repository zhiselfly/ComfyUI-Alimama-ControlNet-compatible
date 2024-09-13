import comfy
import torch
import comfy.ldm.modules.diffusionmodules.mmdit

class AlimamaControlNet(comfy.cldm.mmdit.ControlNet):
    def __init__(
        self,
        num_blocks = None,
        dtype = None,
        device = None,
        operations = None,
        extra_cond_channels = 0,
        **kwargs,
    ):        
        super(comfy.cldm.mmdit.ControlNet, self).__init__(dtype=dtype, device=device, operations=operations, final_layer=False, num_blocks=num_blocks, **kwargs)

        self.extra_cond_channels = extra_cond_channels
        # controlnet_blocks
        self.controlnet_blocks = torch.nn.ModuleList([])
        for _ in range(len(self.joint_blocks)):
            self.controlnet_blocks.append(operations.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype))

        self.pos_embed_input = comfy.ldm.modules.diffusionmodules.mmdit.PatchEmbed(
            None,
            self.patch_size,
            self.in_channels + extra_cond_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=False,
            dtype=dtype,
            device=device,
            operations=operations
        )
