"""English-Tamil parallel text corpus"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import os


_CITATION = """\
                                       Dataset 1
@inproceedings {
    biblio:RaBoMorphologicalProcessing2012,
	title = {Morphological Processing for English-Tamil Statistical Machine Translation},
	author = {Loganathan Ramasamy and Ond{\v{r}}ej Bojar and Zden{\v{e}}k {\v{Z}}abokrtsk{\'{y}}},
	year = {2012},
	pages = {113--122},
	Booktitle = {Proceedings of the Workshop on Machine Translation and Parsing in Indian Languages ({MTPIL}-2012)},
}

                                       Dataset 2
                                       (OPUS:- collection of many OpenSubtitles corpus)									   
Corpus Name: Tanzil 
     Package: Tanzil in Moses format
     Website: http://opus.nlpl.eu/Tanzil-v1.php
Corpus Name: Ubuntu
     Package: Ubuntu.en-ta_LK in Moses format
     Website: http://opus.nlpl.eu/Ubuntu-v14.10.php
Corpus Name: GNOME
     Package: GNOME.en-ta in Moses format
     Website: http://opus.nlpl.eu/GNOME-v1.php
Corpus Name: KDE4
     Package: KDE4.en-ta in Moses format
     Website: http://opus.nlpl.eu/KDE4-v2.php
Corpus Name: OpenSubtitles
     Package: OpenSubtitles in Moses format
     Website: http://opus.nlpl.eu/OpenSubtitles-v2018.php
	 

                                        Dataset 3
										https://github.com/joshua-decoder/indian-parallel-corpora/blob/master/LICENSE
										
"""
_DESCRIPTION = """\
This corpus is a collection of english and tamil parallel text extracted 3 sources (mentioned above as dataset 1, dataset 2, dataset 3).
All the three were merged into a single file and cleaned.The various data cleaning steps were
a)Removed duplicates lines
b)Removed lines containing html tags
c) Removed empty lines and lines which contains an empty english or tamil translation
d) Removed short english transciptions with word length len(word) <= 2


After cleaning there are total of 380245 lines in the corpus.All the lines which were removed were collected in a separate text file
Script used for data cleaning :- https://colab.research.google.com/drive/1AViiS_4-ClngwZ4bKYbmlq3cVAnNHCk0#scrollTo=MWn-AuohbfXI

Note:-
For dataset-3, the data was created by non-expert translators hired over Mechanical Turk and other 
datasets were created from extracting text from movie subtitles, Bible, Quran, cinema, news so the quality of translations are of mixed.
However, it should be useful enough to get you started training models.


Reference for dataset 1 :- http://ufal.mff.cuni.cz/~ramasamy/parallel/html/
Reference for dataset 2 :- http://opus.nlpl.eu/
Reference for dataset 3 :- https://github.com/joshua-decoder/indian-parallel-corpora

"""
test_file_link    = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_test.tar.gz'
valid_file_link    = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_valid.tar.gz'
train_file_1_link = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_train_split1.tar.gz'
train_file_2_link = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_train_split2.tar.gz'
train_file_3_link = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_train_split3.tar.gz'

class EnTamParallelText(tfds.core.GeneratorBasedBuilder):
  """TODO(en_tam_parallel_text): Short description of my dataset."""

  # TODO(en_tam_parallel_text): Set up version.
  VERSION = tfds.core.Version('0.1.0')
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self, description=_DESCRIPTION, features=tfds.features.Translation(languages=("input", "target"), encoder_config=tfds.features.text.TextEncoderConfig()), urls=["http://ufal.mff.cuni.cz/~ramasamy/parallel/html/", "http://opus.nlpl.eu/", "https://github.com/joshua-decoder/indian-parallel-corpora"], \
        supervised_keys=("input", "target"), citation=_CITATION,)

  def _split_generators(self, dl_manager):
    """Load the data from the manual_dir"""
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
	# download link ='https://drive.google.com/uc?export=download&confirm=WCqK&id=12kSyzoUZo8k4BYoU-LzqkfS0-rfw-i4U'
	# downloaded file_name =en_tam_parrallel_text_nodups_sorted_eng_tam_reversed.zip
	#copy path :-  usr/local/lib/python3.6/dist-packages/tensorflow_datasets/translate/en_tam_parallel_text.py
	#checksum path :- usr/local/lib/python3.6/dist-packages/tensorflow_datasets/url_checksums
	# Return the data as 3 splits.
    extracted_path_test = dl_manager.download_and_extract(test_file_link)
    extracted_path_valid = dl_manager.download_and_extract(valid_file_link)
    extracted_path_train = dl_manager.download([train_file_1_link,train_file_2_link,train_file_3_link])
    print(extracted_path_train)
    os.system('''cat /root/tensorflow_datasets/downloads/prav_Neur-Mach-Tran-Engl-Tami-mode_raw_QN0es9cN5Og4Oh6LhrJUFwGFCRjJFq-ildhT1aOl7Is.tar.gz /root/tensorflow_datasets/downloads/prav_Neur-Mach-Tran-Engl-Tami-mode_raw_7Ie2jTwRwTzVN8r3CMLB5wYfyT09qPxpTXzdrZSWZlw.tar.gz /root/tensorflow_datasets/downloads/prav_Neur-Mach-Tran-Engl-Tami-mode_raw_ZTsGeUIhPaKLW_iPEU2U9gUx8rtnSKH2WzfvSfTcEIQ.tar.gz > /root/tensorflow_datasets/downloads/en_tam_parallel_corpus_train.tar.gz''')
    #print(os.path.exists("../root/tensorflow_datasets/downloads/en_tam_parallel_corpus_train.tar.gz"),'********************************')
    extracted_path_train = dl_manager.extract("../root/tensorflow_datasets/downloads/en_tam_parallel_corpus_train.tar.gz")
    return [
          tfds.core.SplitGenerator(name=tfds.Split.TRAIN, num_shards=2,\
              gen_kwargs={"data_file":extracted_path_train}),
          tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, num_shards=1,\
              gen_kwargs={"data_file":extracted_path_valid}),
        #os.path.join(extracted_path, "en_tam_parallel_corpus_validation")}),\
          tfds.core.SplitGenerator(name=tfds.Split.TEST, num_shards=1,\
              gen_kwargs={"data_file":extracted_path_test})
        #os.path.join(extracted_path_test, "en_tam_parallel_corpus_test")})
      ]

  def _generate_examples(self, data_file):
    with tf.io.gfile.GFile(data_file) as dataset:
      for line in dataset:
        line_parts = line.strip().split('\t')
        ip, target = line_parts[0].strip(), line_parts[1].strip()
        yield {"input": ip, "target": target}
		