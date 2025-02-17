import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from datasets import DATASETS
from helpers.bbox_tokenizer import KEY_MULT_DICT, BboxTokenizer


logger = logging.getLogger(__name__)

SPECIAL_TOKEN_VOCABULARIES = ["pad", "bos", "eos", "sep", "mask"]
CHOICES = {
    "shared_bbox_vocab": ["xywh", "x-y-w-h"],
    "var_order": ["c-x-y-w-h", "c-w-h-x-y"],
    # "bbox_quantization": ["linear", "kmeans"],
    "bbox_quantization": ["linear", "kmeans", "percentile"],
}


def _pad_sequence(seq: LongTensor, max_seq_length: int, value: Any) -> LongTensor:
    S = seq.shape[0]
    new_shape = list(seq.shape)
    s = max_seq_length - S
    if s > 0:
        new_shape[0] = s
        pad = torch.full(new_shape, value, dtype=seq.dtype)
        new_seq = torch.cat([seq, pad], dim=0)
    else:
        new_seq = seq

    return new_seq


class LayoutTokenizer:
    """
    Tokenizer converts inputs into (dict of) a sequence
    This is a base class for all tokenizers
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        dataset_cfg: DictConfig,
    ) -> None:
        self._data_cfg = data_cfg
        self._dataset_cfg = dataset_cfg

        name = dataset_cfg._target_.split(".")[-1]
        inv_dic = {str(v.__name__): k for (k, v) in DATASETS.items()}

        # validation
        self._var_order = data_cfg.get("var_order", "c-x-y-w-h")
        # assert self.var_order in ["c-x-y-w-h", "c-w-h-x-y", "c-xw-yh", "c-xywh"]
        assert self._var_order[0] == "c"
        assert all(token in SPECIAL_TOKEN_VOCABULARIES for token in self.special_tokens)
        if "mask" in self.special_tokens:
            assert self.special_tokens.index("mask") == self.N_sp_token - 1

        dataset_name = f"{inv_dic[name]}"
        self._bbox_tokenizer = BboxTokenizer(
            num_bin_bboxes=data_cfg.num_bin_bboxes,
            var_order=self._var_order,
            shared_bbox_vocab=data_cfg.get("shared_bbox_vocab", "xywh"),
            bbox_quantization=data_cfg.get("bbox_quantization", "linear"),
            dataset_name=dataset_name,
        )

        self._N_category = len(DATASETS[inv_dic[name]].labels)

        logger.info(
            f"N_total={self.N_total}, (N_label, N_bbox, N_sp_token)=({self.N_category},{self.N_bbox},{self.N_sp_token})"
        )

        self._special_token_name_to_id = {
            token: self.special_tokens.index(token) + self.N_category + self.N_bbox
            for token in self.special_tokens
        }
        self._special_token_id_to_name = {
            v: k for (k, v) in self._special_token_name_to_id.items()
        }

    def _pad_until(
        self, label: LongTensor, bbox: FloatTensor, mask: BoolTensor
    ) -> Tuple[LongTensor, FloatTensor, BoolTensor]:
        if self.pad_until_max:
            label = _pad_sequence(label, self.max_seq_length, 0)
            bbox = _pad_sequence(bbox, self.max_seq_length, 0)
            mask = _pad_sequence(mask, self.max_seq_length, False)
        return label, bbox, mask

    def _fix_padded_sequences(
        self, label: LongTensor, bbox: FloatTensor, mask: BoolTensor
    ) -> Tuple[LongTensor, FloatTensor]:
        pad_mask = ~mask
        if "pad" in self.special_tokens:
            pad_id = self.name_to_id("pad")
            label[pad_mask] = pad_id
            bbox[pad_mask] = pad_id
        return label, bbox

    def _filter_invalid_labels_and_bboxes(
        self, label: LongTensor, bbox: FloatTensor
    ) -> BoolTensor:
        # If a set of tokens for an element is corrupted, discard the element
        label_valid = (0 <= label) & (label < self.N_category)
        bbox_valid = (0 <= bbox) & (bbox < self.N_bbox)
        bbox_valid = torch.all(bbox_valid, dim=-1)
        invalid = torch.logical_not(label_valid & bbox_valid)
        return invalid

    def _filter_eos(self, label: LongTensor) -> BoolTensor:
        if "bos" in self.special_tokens and "eos" in self.special_tokens:
            invalid = torch.cumsum(label == self.name_to_id("eos"), dim=1) > 0
        else:
            invalid = torch.full(label.size(), fill_value=False)
        return invalid

    @property
    def bbox_tokenizer(self) -> BboxTokenizer:
        return self._bbox_tokenizer

    @property
    def max_seq_length(self) -> int:
        return self._dataset_cfg.max_seq_length

    @property
    def max_token_length(self) -> int:
        return self.max_seq_length * 2 * self.N_var_per_element + 4 \
                if "bos" in self.special_tokens and "eos" in self.special_tokens \
                else self.max_seq_length * 2 * self.N_var_per_element

    @property
    def N_bbox(self) -> int:
        return self.bbox_tokenizer.bbox_vocab_len

    @property
    def N_bbox_per_var(self) -> int:
        return self.bbox_tokenizer.num_bin_bboxes

    @property
    def N_category(self) -> int:
        return self._N_category

    @property
    def N_sp_token(self) -> int:
        return len(self.special_tokens)

    @property
    def N_total(self) -> int:
        return self.N_category + self.N_bbox + self.N_sp_token

    @property
    def N_var_per_element(self) -> int:
        return len(self.var_names)

    @property
    def special_tokens(self) -> List[str]:
        return self._data_cfg.special_tokens

    @property
    def pad_until_max(self) -> bool:
        return self._data_cfg.pad_until_max

    @property
    def var_names(self) -> List[str]:
        return self._var_order.split("-")

    @property
    def var_order(self) -> str:
        return self._var_order

    # functions below are for accesing special token properties
    def name_to_id(self, name: str) -> int:
        return self._special_token_name_to_id[name]

    def id_to_name(self, id_: int) -> str:
        return self._special_token_id_to_name[id_]


class LayoutPairSequenceTokenizer(LayoutTokenizer):
    def __init__(
        self,
        data_cfg: DictConfig,
        dataset_cfg: DictConfig,
    ) -> None:
        super().__init__(data_cfg, dataset_cfg)

    def _filter_special_tokens(self, token_ids: LongTensor) -> LongTensor:
        sep_index = (token_ids == self.name_to_id("sep")).nonzero(as_tuple=True)[0][0]
        token_ids = token_ids[sep_index+1:]

        if self.pad_until_max:
            pad_index = (token_ids == self.name_to_id("pad")).nonzero(as_tuple=True)[0][0]
            token_ids = token_ids[:pad_index]

        if "bos" in self.special_tokens and "eos" in self.special_tokens:
            token_ids = token_ids[1:-1]

        return token_ids

    def encode(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        all_label = deepcopy(rearrange(inputs["label"], "b s -> b s 1"))
        all_mask = deepcopy(inputs["mask"])

        all_input_bbox = deepcopy(self.bbox_tokenizer.encode(inputs["input_bbox"]))
        all_output_bbox = deepcopy(self.bbox_tokenizer.encode(inputs["output_bbox"]))

        all_input_bbox += self.N_category
        all_output_bbox += self.N_category

        B, S = all_label.size()[:2]
        C = self.N_var_per_element

        seq_len = reduce(all_mask.int(), "b s -> b 1", reduction="sum")
        indices = rearrange(torch.arange(0, S), "s -> 1 s")
        assert torch.all(torch.logical_not(all_mask) == (seq_len <= indices)).item()

        all_seq, all_masked_seq, all_masking = [], [], []
        for i in range(B):
            mask = all_mask[i]
            label = all_label[i][mask]

            input_bbox = all_input_bbox[i][mask]
            output_bbox = all_output_bbox[i][mask]
            mask = mask[mask]

            input_seq = torch.cat([label, input_bbox], axis=-1)
            input_seq = rearrange(input_seq, "s x -> (s x)")
            output_seq = torch.cat([label, output_bbox], axis=-1)

            masked_output_seq = deepcopy(output_seq)
            masked_output_seq[..., 1:] = self.name_to_id("mask")
            masked_output_seq[0] = output_seq[0]

            output_seq = rearrange(output_seq, "s x -> (s x)")
            masked_output_seq = rearrange(masked_output_seq, "s x -> (s x)")

            mask = repeat(mask, "s -> (s c)", c=C*2)

            if "bos" in self.special_tokens and "eos" in self.special_tokens:
                bos = torch.tensor([self.name_to_id("bos")])
                eos = torch.tensor([self.name_to_id("eos")])

                input_seq = torch.cat([bos, input_seq, eos], axis=-1)
                output_seq = torch.cat([bos, output_seq, eos], axis=-1)
                masked_output_seq = torch.cat([bos, masked_output_seq, eos], axis=-1)

                mask = torch.cat([torch.full((4,), fill_value=True), mask], axis=-1)

            sep = torch.tensor([self.name_to_id("sep")])
            seq = torch.cat([input_seq, sep, output_seq], axis=-1)
            masked_seq = torch.cat([input_seq, sep, masked_output_seq], axis=-1)

            mask = torch.cat([torch.full((1,), fill_value=True), mask], axis=-1)
            mask[masked_seq == self.name_to_id("mask")] = False

            if self.pad_until_max:
                seq = _pad_sequence(seq, self.max_token_length, self.name_to_id("pad"))
                masked_seq = _pad_sequence(masked_seq, self.max_token_length, self.name_to_id("pad"))
                mask = _pad_sequence(mask, self.max_token_length, True)

            all_seq.append(seq)
            all_masked_seq.append(masked_seq)
            all_masking.append(mask)
        
        all_seq = torch.stack(all_seq, dim=0)
        all_masked_seq = torch.stack(all_masked_seq, dim=0)
        all_masking = torch.stack(all_masking, dim=0)

        return {"seq": all_seq.long(), "masked_seq": all_masked_seq.long(), "mask": all_masking}

    def decode(self, all_seq: LongTensor) -> Dict[str, Tensor]:
        B, S = all_seq.size()[:2]

        all_bbox, all_label, all_mask = [], [], []
        for i in range(B):
            seq = all_seq[i]
            seq = self._filter_special_tokens(seq)
            seq = rearrange(seq, "(s c) -> s c", c=self.N_var_per_element)[1:, :]  ### remove canvas

            label, bbox = deepcopy(seq[:, 0]), deepcopy(seq[:, 1:])
            bbox -= self.N_category
            mask = torch.ones(label.size(), dtype=bool)

            bbox = _pad_sequence(bbox, self.max_seq_length, 0.0)
            label = _pad_sequence(label, self.max_seq_length, 0)
            mask = _pad_sequence(mask, self.max_seq_length, False)

            all_bbox.append(bbox)
            all_label.append(label)
            all_mask.append(mask)

        all_bbox = torch.stack(all_bbox, dim=0)
        all_label = torch.stack(all_label, dim=0)
        all_mask = torch.stack(all_mask, dim=0)

        invalid = self._filter_invalid_labels_and_bboxes(all_label, all_bbox)
        all_bbox = self.bbox_tokenizer.decode(all_bbox)

        all_label[invalid] = 0
        all_bbox[invalid] = 0.0

        all_bbox[~all_mask] = 0.0

        return {"bbox": all_bbox, "label": all_label, "mask": all_mask}

    def token_mask(self, seq_input: LongTensor) -> BoolTensor:
        masks = self.bbox_tokenizer.token_mask
        last = BoolTensor(
            [True if x == "sep" else False for x in self.special_tokens]
        )

        masks["c"] = torch.cat(
            [
                torch.full((self.N_category,), True),
                torch.full((self.N_bbox,), False),
                last,
            ]
        )
        for key in self.var_names:
            if key == "c":
                continue
            masks[key] = torch.cat(
                [torch.full((self.N_category,), False), masks[key], last]
            )
        attr_mask = torch.stack([masks[k] for k in self.var_names], dim=0)

        sep_index = torch.nonzero(seq_input == self.name_to_id("sep"))[:, 1][0]
        sep_index += 1 if "bos" in self.special_tokens and "eos" in self.special_tokens else sep_index
        print(int(sep_index))

        mask = repeat(attr_mask, "x c -> (s x) c", s=self.max_token_length)
        mask = mask[:self.max_token_length, :]

        back_mask = repeat(attr_mask, "x c -> (s x) c", s=self.max_token_length)
        mask[sep_index:, :] = back_mask[:self.max_token_length - sep_index, :]

        return mask