"""
Adapted from: https://github.com/azshue/TPT/blob/main/clip/custom_clip.py
Paper: https://arxiv.org/pdf/2209.07511.pdf
"""

import torch
import torch.nn as nn
import logging

from open_clip import create_model_and_transforms, get_tokenizer
from datasets.cls_names import get_class_names

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, arch_name, class_names, n_ctx=16, ctx_init=None, class_token_pos='end', learned_cls=False):
        super().__init__()
        self.n_cls = len(class_names)
        self.learned_cls = learned_cls
        self.class_names = class_names

        self.dtype = clip_model.visual.conv1.weight.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        self.tokenize = get_tokenizer(arch_name)

        if ctx_init:
            # use given words to initialize context vectors
            logger.info("Initializing the context with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                class_token_pos = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            logger.info("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.class_token_position = class_token_pos

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # setup the rest using the specified class names
        self.reset_class_names(class_names)

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_class_names(self, class_names):
        self.n_cls = len(class_names)
        if not self.learned_cls:
            class_names = [name.replace("_", " ") for name in class_names]
            name_lens = [len(self.tokenize(name)) for name in class_names]
            prompts = [self.prompt_prefix + " " + name + "." for name in class_names]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in class_names]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in class_names]

            # TODO: re-init the cls parameters
            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        with torch.no_grad():
            tokenized_prompts = torch.cat([self.tokenize(p) for p in prompts]).to(self.device)
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.class_names = class_names

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        ctx = init if init is not None else self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.learned_cls:
            assert self.class_token_position == "end"

        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(f"Class token position '{self.class_token_position}' is not supported."
                             f" Choose from: end, middle, front")

        return prompts


class ClipTestTimePromptTuning(nn.Module):
    def __init__(self, clip_model, normalization, arch_name, dataset_name, n_ctx=16,
                 ctx_init=None, class_token_pos='end', learned_cls=False):
        super(ClipTestTimePromptTuning, self).__init__()

        # setup the underlying CLIP model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.data
        self.normalize = normalization

        # get the class names form the dataset name
        class_names = get_class_names(dataset_name)

        # prompt tuning
        self.prompt_learner = PromptLearner(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_class_names(self, class_names):
        self.prompt_learner.reset_class_names(class_names)

    def get_text_features(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, image):
        image = self.normalize(image.type(self.dtype))
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.get_text_features()
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits
