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

import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
from PIL import Image
from test.util import get_evadb_for_testing
from evadb.server.command_handler import execute_query_fetch_all


class ImageGenerationFunctionTest(unittest.TestCase):
    def setUp(self):
        self.evadb = get_evadb_for_testing()
        self.evadb.catalog().reset()
        execute_query_fetch_all(self.evadb, """CREATE TABLE IF NOT EXISTS ImageGenerator (prompt TEXT(100));""")
        execute_query_fetch_all(self.evadb, """INSERT INTO ImageGenerator (prompt) VALUES ('a city at night')""")

    def tearDown(self) -> None:
        execute_query_fetch_all(self.evadb, "DROP TABLE IF EXISTS ImageGenrator;")

    @patch.dict("os.environ", {"OPENAI_KEY": "mocked_openai_key"})
    @patch("requests.get")
    @patch("openai.Image.create", return_value={"data": [{"url": "mocked_url"}]})
    def test_image_generation(self, mock_openai_create, mock_requests_get):
        # Generate a 1x1 white pixel PNG image in memory
        img = Image.new("RGB", (1, 1), color="white")
        img_byte_array = BytesIO()
        img.save(img_byte_array, format="PNG")
        mock_image_content = img_byte_array.getvalue()
        mock_response = MagicMock()
        mock_response.content = mock_image_content
        mock_requests_get.return_value = mock_response

        function_name = "ImageGeneration"
        execute_query_fetch_all(self.evadb, f"DROP FUNCTION IF EXISTS {function_name};")

        create_function_query = f"""CREATE FUNCTION IF NOT EXISTS {function_name}
            IMPL 'evadb/functions/image_generation.py';
        """
        execute_query_fetch_all(self.evadb, create_function_query)

        gpt_query = f"SELECT {function_name}(prompt) FROM ImageGenrator;"
        execute_query_fetch_all(self.evadb, gpt_query)

        mock_openai_create.assert_called_once_with(
            prompt="a city at night", n=1, size="1024x1024"
        )
