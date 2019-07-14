"""Tests for English-Tamil parallel dataset module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.translate import en_tam_parallel_text

#Overlapping lines in test , train       (3 lines overlapping)
#Overlapping lines in test , validation  (1 line overlapping)
#Not able to remove the overlapping lines since only the hashvalue of the lines are thrown out in the Traceback  but not the actual line or its index 
#Acutal count of lines of text {"test" : 26000, "train" : 324196, "validation" : 25245}

class EnTamParallelTextTest(testing.DatasetBuilderTestCase):
  DATASET_CLASS = en_tam_parallel_text.EnTamParallelText
  #OVERLAPPING_SPLITS = ['test', 'validation']
  SPLITS = {"test" : 2, "train" : 2, "validation" : 2}



if __name__ == "__main__":
  testing.test_main()
  