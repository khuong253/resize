import copy
import logging
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from data.util import sparse_to_dense
from helpers.layout_tokenizer import LayoutPairSequenceTokenizer
from helpers.sampling import sample
from helpers.task import duplicate
from helpers.util import batch_shuffle_index
from models.base_model import BaseModel
from models.common.nn_lib import (
    CategoricalEncDecTransformer,
    CustomDataParallel,
)
from models.common.util import get_dim_model, shrink

logger = logging.getLogger(__name__)


class BART(BaseModel):
    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutPairSequenceTokenizer,
        tasks: Union[str, List[str]] = [],  # ["random"], ["c", "cwh", "partial"],
        pos_emb: str = "default",  # "default"
    ) -> None:
        super().__init__()

        kwargs = {}
        if pos_emb == "elem_attr":
            kwargs["n_attr_per_elem"] = tokenizer.N_var_per_element

        self.tokenizer = tokenizer
        assert self.tokenizer.var_order == "c-w-h-x-y"
        # assert self.tokenizer.special_tokens == ["pad", "bos", "eos", "sep", "mask"]

        # Note: make sure learnable parameters are inside self.model
        # backbone = instantiate(backbone_cfg)
        # dim_model=get_dim_model(backbone_cfg)

        # backbone_enc_cfg = shrink(backbone_cfg, 21 / 32)
        # backbone_dec_cfg = shrink(backbone_cfg, 21 / 32)

        backbone_enc_cfg = backbone_cfg
        backbone_dec_cfg = backbone_cfg

        backbone_enc = instantiate(backbone_enc_cfg)

        params = {
            k: v
            for (k, v) in backbone_dec_cfg["encoder_layer"].items()
            if k != "_target_"
        }
        backbone_dec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(**params),
            num_layers=backbone_dec_cfg["num_layers"],
        )

        self.model = CustomDataParallel(
            CategoricalEncDecTransformer(
                backbone_enc=backbone_enc,
                backbone_dec=backbone_dec,
                dim_model=get_dim_model(backbone_enc_cfg),
                num_classes_dec=self.tokenizer.N_total,
                max_token_length_dec=self.tokenizer.max_token_length,
                pos_emb=pos_emb,
                **kwargs,
            )
        )
        self.apply(self._init_weights)
        self.compute_stats()
        self.loss_fn_ce = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.name_to_id("pad")
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        # outputs = self.model(input=inputs["input"], target=inputs["target"][:, :-1])
        outputs = self.model(input=inputs["input"], target=inputs["target"][:, 1:])
        nll_loss = self.loss_fn_ce(
            rearrange(outputs["logits"], "b s c -> b c s"),
            inputs["target"][:, 1:],
        )
        losses = {"nll_loss": nll_loss}
        outputs["ids"] = torch.argmax(outputs["logits"], dim=-1)
        return outputs, losses

    def sample(
        self,
        batch_size: Optional[int],
        inputs: Optional[Tensor] = None,
        sampling_cfg: Optional[DictConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:

        input_seq = inputs["masked_seq"]
        mask_user = inputs["mask"].clone()

        enc_input = duplicate(input_seq, batch_size)
        dec_input = enc_input.clone()[:, 0].unsqueeze(1)

        B, S = batch_size, input_seq.size(1)

        for i in range(S):
            print(i)
            logits = self.model(enc_input.to(device), dec_input.to(device))[
                "logits"
            ].cpu()
            logits = rearrange(logits[:, i : i + 1], "b 1 c -> b c")
          
            # constrained decoding
            token_mask = self.tokenizer.token_mask(seq_input=input_seq)
            invalid = repeat(~token_mask[i : i + 1], "1 c -> b c", b=B)
            logits[invalid] = -float("Inf")

            print(invalid)
            print(logits)

            predicted = sample(logits, sampling_cfg)
  
            id_ = enc_input[:, i + 1 : (i + 1) + 1]
            flag = ~mask_user[:, i + 1 : (i + 1) + 1]
            if id_.size(1) == 1:
                predicted = torch.where(flag, predicted, id_)

            dec_input = torch.cat([dec_input, predicted], dim=1)
            print(dec_input[0])

        seq = dec_input.clone().cpu() 
        layouts = self.tokenizer.decode(seq)
        return layouts

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        data = self.tokenizer.encode({ "input_bbox": bbox["input_bbox"], "output_bbox": bbox["output_bbox"], "label": label, "mask": mask})

        return {
            "target": data["seq"],
            "input": data["masked_seq"],
        }

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        additional_no_decay = []
        for base in ["input_pos_emb", "target_pos_emb"]:
            for n in getattr(self.model.module, base).no_decay_param_names:
                additional_no_decay.append(f"model.module.{base}.{n}")
        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )