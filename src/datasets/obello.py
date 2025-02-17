import os

import torch
from fsspec.core import url_to_fs
from torch_geometric.data import Data
import json

from datasets.base import BaseDataset


class ObelloDataset(BaseDataset):
    name = "obello"
    labels = [
        "canvas",
        "image",
        "logo",
        "headline",
        "body",
        "cta",
        "graphicShape",
        "line"
    ]

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)

    def download(self):
        # super().download()
        pass

    def process(self):
        fs, _ = url_to_fs(self.raw_dir)
        
        for split_publaynet in ["train", "test"]:
            json_path = os.path.join(self.raw_dir, f"{split_publaynet}.json")
            raw_data = json.load(open(json_path, 'r'))
            
            data_list = []
            for paired_layout in raw_data:
                input_layout, output_layout = paired_layout["Input layout"], paired_layout["Output layout"]
                input_size, output_size = [input_layout["width"], input_layout["height"]], [output_layout["width"], output_layout["height"]]
                
                group_id = input_layout["groupId"]
                size_type = f"{input_size[0]}x{input_size[1]}-{output_size[0]}x{output_size[1]}"

                elements = input_layout['elements']
                N = len(elements)
                if N == 0 or N > self.max_seq_length:
                    continue

                input_bbox = [[0, 0, input_layout["width"], input_layout["height"]]]
                input_bbox += [[e["x"], e["y"], e["width"], e["height"]] for e in input_layout["elements"]]
                input_bbox = torch.tensor(input_bbox, dtype=torch.float)

                output_bbox = [[0, 0, output_layout["width"], output_layout["height"]]]
                output_bbox += [[e["x"], e["y"], e["width"], e["height"]] for e in output_layout["elements"]]
                output_bbox = torch.tensor(output_bbox, dtype=torch.float)

                label = [self.label2index["canvas"]]
                label += [self.label2index[e["class"]] for e in input_layout["elements"]]
                label = torch.tensor(label, dtype=torch.int)

                bbox = torch.cat((input_bbox, output_bbox), dim=1)

                data = Data(x=bbox, y=label)
                data.attr = {
                    "group_id": group_id,
                    "size_type": size_type,
                }
                data_list.append(data)

            if split_publaynet == "train":
                train_list = data_list
            else:
                test_list = data_list

        # shuffle train with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(train_list), generator=generator)
        train_list = [train_list[i] for i in indices]

        with fs.open(self.processed_paths[0], "wb") as file_obj:
            torch.save(self.collate(train_list), file_obj)
        with fs.open(self.processed_paths[1], "wb") as file_obj:
            torch.save(self.collate(test_list), file_obj)