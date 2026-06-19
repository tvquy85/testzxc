import pandas as pd
import datasets
import os
import logging

# 数据集路径设置
META_DATA_PATH_TRAIN = 'xxx/dataset/tulu2/smac/ppo_train.json' 
META_DATA_PATH_TEST ='xxx/dataset/tulu2/smac/ppo_test.json'

_FEATURES = datasets.Features(
    {
        "prompt": datasets.Value("string"),      
        "gt_action": datasets.Value("int32"),
       
    },
)


class create_data(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default", version=datasets.Version("0.0.2"))]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="None",
            features=_FEATURES,
            supervised_keys=None,
            homepage="None",
            license="None",
            citation="None",
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": META_DATA_PATH_TRAIN
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": META_DATA_PATH_TEST
                },
            ),
        ]

    def _generate_examples(self, metadata_path):
        
        metadata = pd.read_json(metadata_path, lines=True)
        
        for _, rrow in metadata.iterrows():
            for i, row in enumerate(rrow):
                prompt = row["prompt"]
              
                gt_action = row["gt_action"]
                
                key = i
               

                yield key, {
                    "prompt": prompt,
                   
                    "gt_action": gt_action
                }

