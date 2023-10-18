# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from evadb.functions.decorators import PyTorchClassifierAbstractFunction, setup, forward
from evadb.functions.abstract.abstract_function import AbstractFunction, setup
from evadb.functions.decorators.decorators import forward
import torch
import requests
import pandas as pd
import numpy as np
from PIL import Image
import os
from io import BytesIO
from typing import List
from evadb.functions.decorators.io_descriptors.data_types import IOColumnArgument, PyTorchTensor, PandasDataframe, NdArrayType, Tensor
from evadb.configuration.configuration_manager import ConfigurationManager
import openai

class ImageGenerationFunction(AbstractFunction):

    @setup(cacheable=True, function_type="image_generation", batchable=True)
    def setup(self) -> None:
        pass

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["prompt"],
                is_nullable=False,
                column_types=[NdArrayType.STR,],
                column_shapes=[(None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["image"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
    )
    def forward(self, text_df):
        openai.organization = "org-NGm7tEKgvz4WKHeaT2YrnAq6"
        openai.api_key = os.getenv("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")

        def generate(text_df: PandasDataframe):
            results = []
            queries = text_df[text_df.columns[0]]
            for query in queries:
                response = openai.Image.create(
                    prompt = query, 
                    n = 1, 
                    size = "1024x1024"
                )   
                image_url = response['data'][0]["url"]
                image_bytes = BytesIO(image_url.content)
                image = Image.open(image_bytes) # get image from the link
                frame = np.array(image) # convert to format to store in data frame
                results.append(frame) # store in data frame

            return results

        df = pd.DataFrame({"image": generate(text_df=text_df)})
        return df


