import torch
import torch.nn as nn 
import timm
from architectures import CLASSIFIERS_ARCHITECTURES, get_architecture

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class DiffusionRobustModel(nn.Module):
    def __init__(self, classifier_name="beit"):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("imagenet/256x256_diffusion_uncond.pt")
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 
        self.classifier_name = classifier_name

        # Load the classifier model
        if classifier_name == "beit":
            classifier = timm.create_model('beit_large_patch16_512', pretrained=True)
            

        elif classifier_name == "vit":
            model_path = "/storage/vatsal/models/hyper/ViT-1e-1/checkpoint.pth.tar"
            classifier = get_architecture("google/vit-base-patch16-224-in21k", "hyper")
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))

        classifier.eval().cuda()
        self.classifier = classifier

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()

    def forward(self, x, t):
        x_in = x * 2 -1
        imgs = self.denoise(x_in, t)

        if self.classifier_name == "vit":
            imgs = torch.nn.functional.interpolate(imgs, (224, 224), mode='bicubic', antialias=True)

        elif self.classifier_name == "beit":
            imgs = torch.nn.functional.interpolate(imgs, (512, 512), mode='bicubic', antialias=True)

        imgs = torch.tensor(imgs).cuda()
        with torch.no_grad():
            out = self.classifier(imgs)

        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out