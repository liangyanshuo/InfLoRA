import torch
import torch.nn as nn
import copy

from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.zoo import L2P

class ViT_Prompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)


    def forward(self, x, register_blk=-1, prompt=None, q=None, train=None, task_id=None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)


        # if instance_tokens is not None:
        #     instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i,blk in enumerate(self.blocks):

            if prompt is not None:
                if train:
                    p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id)
                    prompt_loss += loss
                else:
                    p_list, _, x = prompt.forward(q, i, x, train=False, task_id=task_id)
            else:
                p_list = None

            x = blk(x, register_blk==i, prompt=p_list)


        # if instance_tokens is not None:
        #     x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)

        x = self.norm(x)
        
        return x, prompt_loss



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_Prompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)

        # n = 0
        # for p in self.image_encoder.parameters():
        #     n += p.numel()
        # print(n)
        # print(self.image_encoder)
        # exit()

        self.class_num = 1
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.prompt_pool = L2P(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image):
        image_features, _ = self.image_encoder(image)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, feature):
        logits = []
        image_features, prompt_loss = self.image_encoder(image, q=feature, prompt = self.prompt_pool, train=True, task_id=(self.numtask-1)*torch.ones(len(image)).to(image.device).long())
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

    def interface(self, image, feature):
        image_features, _ = self.image_encoder(image, q=feature, prompt=self.prompt_pool, train=True)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
