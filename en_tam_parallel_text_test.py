"""Tests for English-Tamil parallel dataset module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.translate import en_tam_parallel_text


import os

class EnTamParallelTextTest(testing.DatasetBuilderTestCase):
  DATASET_CLASS = en_tam_parallel_text.EnTamParallelText
  
  SPLITS = {"test" : 2}
  DL_EXTRACT_RESULT = {"test_file_link" : "en_tam_parallel_corpus_train",
                       "valid_file_link": '' ,
                       "train_file_1_link": '' ,
                       "train_file_2_link": '' ,
                       "train_file_3_link":''}


if __name__ == "__main__":
  testing.test_main()
  