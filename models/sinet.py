# from turtle import forward
# import torch
# import torch.nn as nn
# import copy
# import ipdb
# import random
# from torch.nn import functional as F

# from models.vit import VisionTransformer, PatchEmbed, Block, Attention, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn

# def sample_dirichlet(N, alpha):
#     weight = [random.gammavariate(alpha,1) for _ in range(N)]
#     weight = [w/sum(weight) for w in weight]
#     return weight

# class multi_prompt(nn.Linear):
#     def __init__(self,in_features,out_features, N, bias=False):
#         super(multi_prompt,self).__init__(in_features, out_features, bias=bias)
#         self.N = N

#         if self.N > 1:
#             for n in range(self.N):
#                 self.register_parameter('weight_{}'.format(n), nn.Parameter(torch.zeros_like(self.weight), requires_grad=True))
#             self.attention = nn.ModuleList()
#             for _ in range(out_features):
#                 self.attention.append(Attention())

#     def get_prompt(self, train=False):
#         if self.N > 1:
#             if train:
#                 w = sample_dirichlet(self.N, 2*self.N)
#                 w = torch.tensor(w)
#             else:
#                 w = torch.ones(self.N)/self.N
#             # print(w)
#             w = w.cuda()
#             weight_list = [getattr(self, 'weight_{}'.format(n)) for n in range(self.N)]
#             weight = torch.stack(weight_list, dim=-1)
#             weight = torch.matmul(weight, w)
#         else:
#             weight = self.weight
#         return weight

#     def init_prompt(self,weight_list:list):
#         assert self.N == len(weight_list)
#         if self.N > 1:
#             for n in range(self.N):
#                 weight = getattr(self, 'weight_{}'.format(n))
#                 weight.data.copy_(weight_list[n])
#         else:
#             self.weight.data.copy_(weight_list[0])
#         return

# class ViT_Prompts(VisionTransformer):
#     def __init__(
#             self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
#             embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
#             embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

#         super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
#             embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
#             drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
#             embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)

#     def get_token(self, x, instance_tokens=None):
#         x = self.patch_embed(x)
#         x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

#         if instance_tokens is not None:
#             instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

#         x = x + self.pos_embed.to(x.dtype)
#         if instance_tokens is not None:
#             x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)
#         return x

#     # def add_prompt()
#     def forward_(self, x, instance_tokens=None, **kwargs):
#         if instance_tokens == None:
#             x = self.get_token(x, instance_tokens)
#         else:
#             x = self.get_token(x, instance_tokens[0])
#         x = self.pos_drop(x)
#         x = self.blocks(x)
#         x = self.norm(x)
#         if self.global_pool:
#             x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#         x = self.fc_norm(x)
#         return x

#     # def forward(self, x, instance_tokens=None, T_var=None, alpha=None, **kwargs):
#     #     if instance_tokens == None:
#     #         x = self.get_token(x, instance_tokens)
#     #     elif len(instance_tokens) == 1:
#     #         x = self.get_token(x, instance_tokens[0])
#     #     else:
#     #         assert T_var != None
#     #         assert alpha != None
#     #         xs = []
#     #         for instance_tokens_ in instance_tokens:
#     #             xs.append(self.get_token(x, instance_tokens_))
#     #         x = alpha*torch.matmul(xs[0], T_var) + (1-alpha)*xs[1]
#     #         # x = alpha*torch.matmul(xs[0], T_var.t()) + (1-alpha)*xs[1]
#     #         # x = alpha*xs[0] + (1-alpha)*xs[1]
#     #     x = self.pos_drop(x)
#     #     x = self.blocks(x)
#     #     x = self.norm(x)
#     #     if self.global_pool:
#     #         x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#     #     x = self.fc_norm(x)
#     #     return x

#     def forward(self, x, instance_tokens=None, T_var=None, alpha=None, **kwargs):
#         if instance_tokens == None:
#             x = self.get_token(x, instance_tokens)
#         elif len(instance_tokens) == 1:
#             x = self.get_token(x, instance_tokens[0])
#         else:
#             assert T_var != None
#             assert alpha != None
#             # instance_token = alpha*torch.matmul(instance_tokens[0],T_var) + (1-alpha)*instance_tokens[1]
#             instance_token = alpha*torch.matmul(instance_tokens[0],T_var.t()) + (1-alpha)*instance_tokens[1]
#             # instance_token = alpha*instance_tokens[0] + (1-alpha)*instance_tokens[1]

#             x = self.get_token(x,instance_token)
#         x = self.pos_drop(x)
#         x = self.blocks(x)
#         x = self.norm(x)
#         if self.global_pool:
#             x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#         x = self.fc_norm(x)
#         return x

# def _create_vision_transformer(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     # NOTE this extra code to support handling of repr size for in21k pretrained models
#     # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
#     pretrained_cfg = resolve_pretrained_cfg(variant)
#     default_num_classes = pretrained_cfg['num_classes']
#     num_classes = kwargs.get('num_classes', default_num_classes)
#     repr_size = kwargs.pop('representation_size', None)
#     if repr_size is not None and num_classes != default_num_classes:
#         repr_size = None

#     model = build_model_with_cfg(
#         ViT_Prompts, variant, pretrained,
#         pretrained_cfg=pretrained_cfg,
#         representation_size=repr_size,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         pretrained_custom_load='npz' in pretrained_cfg['url'],
#         **kwargs)
#     return model



# class SiNet(nn.Module):

#     def __init__(self, args):
#         super(SiNet, self).__init__()

#         model_kwargs = dict(patch_size=16, embed_dim=args["embd_dim"], depth=12, num_heads=args["num_heads"])
#         # self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
#         self.image_encoder =_create_vision_transformer('vit_tiny_patch16_224', pretrained=True, **model_kwargs)

#         self.class_num = 1
#         if args["dataset"] == "cddb":
#             self.class_num = 2
#             self.classifier_pool = nn.ModuleList([
#                 nn.Linear(args["embd_dim"], self.class_num, bias=True)
#                 for i in range(args["total_sessions"])
#             ])
#         elif args["dataset"] == "domainnet":
#             self.class_num = 345
#             self.classifier_pool = nn.ModuleList([
#                 nn.Linear(args["embd_dim"], self.class_num, bias=True)
#                 for i in range(args["total_sessions"])
#             ])
#         elif args["dataset"] == "core50":
#             self.class_num = 50
#             self.classifier_pool = nn.ModuleList([
#                 nn.Linear(args["embd_dim"], self.class_num, bias=True)
#                 for i in range(args["total_sessions"])
#             ])

#         else:
#             raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

#         # self.prompt_pool = nn.ModuleList([
#         #     multi_prompt(args["embd_dim"], args["prompt_length"], max(i,1), bias=False)
#         #     for i in range(args["total_sessions"])
#         # ])

#         self.prompt_pool = nn.ModuleList([
#             nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
#             for i in range(args["total_sessions"])
#         ])

#         self.numtask = 0
#         self.args = args

#     @property
#     def feature_dim(self):
#         return self.image_encoder.out_dim

#     def extract_vector(self, image):
#         image_features = self.image_encoder(image)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         return image_features

#     def get_input(self,t, train_loader, nselect = 200):
#         activation = []
#         samples = 0
#         with torch.no_grad():
#             for i, (_, x, targets) in enumerate(train_loader):
#                 if samples > nselect: break
#                 x = x.cuda()
#                 x = self.image_encoder.patch_embed(x)
#                 x = torch.cat((self.image_encoder.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

#                 instance_tokens = self.prompt_pool[t].weight.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

#                 x = x + self.image_encoder.pos_embed.to(x.dtype)
#                 x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)
#                 activation.append(x)
#                 samples += len(x)
#         activation = torch.cat(activation)[:nselect]
#         return activation

#     def seq_init_prompt(self,):
#         # ipdb.set_trace()
#         assert self.numtask > 1
#         self.prompt_pool[self.numtask-1].weight.data.copy_(self.prompt_pool[self.numtask-2].weight)
#         self.classifier_pool[self.numtask-1].load_state_dict(self.classifier_pool[self.numtask-2].state_dict())

#     def mean_init_prompt(self,):
#         assert self.numtask > 1
#         weight = torch.stack([self.prompt_pool[i].weight for i in range(self.numtask-1)],dim=0).mean(0)
#         self.prompt_pool[self.numtask-1].weight.data.copy_(weight)

#         weight = torch.stack([self.classifier_pool[i].weight for i in range(self.numtask-1)],dim=0).mean(0)
#         self.classifier_pool[self.numtask-1].weight.data.copy_(weight)
#         bias = torch.stack([self.classifier_pool[i].bias for i in range(self.numtask-1)],dim=0).mean(0)
#         self.classifier_pool[self.numtask-1].bias.data.copy_(bias)

#     # def init(self,):
#     #     assert self.numtask > 1
#     #     if self.numtask == 2:
#     #         self.prompt_pool[self.numtask-1].weight.data.copy_(self.prompt_pool[self.numtask-2].weight)
#     #         self.classifier_pool[self.numtask-1].load_state_dict(self.classifier_pool[self.numtask-2].state_dict())
#     #     else:
#     #         for i in range(self.numtask-1):
#     #             self.prompt_pool[self.numtask-1].weight.data[self.args["embd_dim"]*i:(i+1)*self.args["embd_dim"]].copy_(self.prompt_pool[i].weight.data[:self.args["embd_dim"]])
#     #     weight = torch.stack([self.classifier_pool[i].weight for i in range(self.numtask-1)],dim=0).mean(0)
#     #     self.classifier_pool[self.numtask-1].weight.data.copy_(weight)
#     #     bias = torch.stack([self.classifier_pool[i].bias for i in range(self.numtask-1)],dim=0).mean(0)
#     #     self.classifier_pool[self.numtask-1].bias.data.copy_(bias)
#     #     return

#     def simplex_init_prompt(self,):
#         # if self.numtask>2: ipdb.set_trace()
#         assert self.numtask > 1
#         weight_list = [self.prompt_pool[i].get_prompt() for i in range(self.numtask-1)]
#         self.prompt_pool[self.numtask-1].init_prompt(weight_list)

#         weight = torch.stack([self.classifier_pool[i].weight for i in range(self.numtask-1)],dim=0).mean(0)
#         self.classifier_pool[self.numtask-1].weight.data.copy_(weight)
#         bias = torch.stack([self.classifier_pool[i].bias for i in range(self.numtask-1)],dim=0).mean(0)
#         self.classifier_pool[self.numtask-1].bias.data.copy_(bias)
#         return

#     def forward(self, image, train=False, alpha=None):
#         logits = []
#         # image_features = self.image_encoder(image, [self.prompt_pool[self.numtask-1].weight])
#         if hasattr(self,'T_var_{}'.format(self.numtask-1)):
#             prompts = [self.prompt_pool[self.numtask-2].weight,self.prompt_pool[self.numtask-1].weight]
#             T_var = getattr(self, 'T_var_{}'.format(self.numtask-1))
#         else:
#             prompts = [self.prompt_pool[self.numtask-1].weight]
#             T_var = None
#         image_features = self.image_encoder(image, prompts, T_var, alpha)
#         for prompts in [self.classifier_pool[self.numtask-1]]:
#             logits.append(prompts(image_features))

#         return {
#             'logits': torch.cat(logits, dim=1),
#             'features': image_features
#         }

#     def interface(self, image, selection):
#         instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
#         image_features = self.image_encoder(image, [instance_batch])

#         logits = []
#         for prompt in self.classifier_pool:
#             logits.append(prompt(image_features))

#         logits = torch.cat(logits,1)
#         selectedlogit = []
#         for idx, ii in enumerate(selection):
#             selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
#         selectedlogit = torch.stack(selectedlogit)
#         return selectedlogit

#     def update_fc(self, nb_classes):
#         self.numtask +=1

#     def copy(self):
#         return copy.deepcopy(self)

#     def freeze(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         self.eval()

#         return self




import torch
import torch.nn as nn
import copy

from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn

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


    def forward(self, x, register_blk=-1, prompt=None, instance_tokens=None, q=None, train=None, task_id=None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)


        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = x + self.pos_embed.to(x.dtype)

        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
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


        if instance_tokens is not None:
            x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)

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

        self.class_num = 1
        if args["dataset"] == "cifar100":
            self.class_num = 10
            self.classifier_pool = nn.ModuleList([
                nn.Linear(args["embd_dim"], self.class_num, bias=True)
                for i in range(args["total_sessions"])
            ])
        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        self.prompt_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
            for i in range(args["total_sessions"])
        ])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image):
        logits = []
        image_features = self.image_encoder(image, self.prompt_pool[self.numtask-1].weight)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, selection):
        instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        image_features = self.image_encoder(image, instance_batch)

        logits = []
        for prompt in self.classifier_pool:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        selectedlogit = []
        for idx, ii in enumerate(selection):
            selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit

    def update_fc(self, nb_classes):
        self.numtask +=1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
