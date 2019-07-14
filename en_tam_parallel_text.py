"""English-Tamil parallel text corpus"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import tempfile
import shutil


_CITATION = """\  
@inproceedings {biblio:RaBoMorphologicalProcessing2012,
	title = {Morphological Processing for English-Tamil Statistical Machine Translation},
	author = {Loganathan Ramasamy and Ond{\v{r}}ej Bojar and Zden{\v{e}}k {\v{Z}}abokrtsk{\'{y}}},
	year = {2012},
	pages = {113--122},
	Booktitle = {Proceedings of the Workshop on Machine Translation and Parsing in Indian Languages ({MTPIL}-2012)},
}

@InProceedings{TIEDEMANN12.463,
  author = {J�rg Tiedemann},
  title = {Parallel Data, Tools and Interfaces in OPUS},
  booktitle = {Proceedings of the Eight International Conference on Language Resources and Evaluation (LREC'12)},
  year = {2012},
  month = {may},
  date = {23-25},
  address = {Istanbul, Turkey},
  editor = {Nicoletta Calzolari (Conference Chair) and Khalid Choukri and Thierry Declerck and Mehmet Ugur Dogan and Bente Maegaard and Joseph Mariani and Jan Odijk and Stelios Piperidis},
  publisher = {European Language Resources Association (ELRA)},
  isbn = {978-2-9517408-7-7},
  language = {english}
 }	 
                                        
https://github.com/joshua-decoder/indian-parallel-corpora/blob/master/LICENSE
										
"""
_DESCRIPTION = """\
This corpus is a collection of english and tamil parallel text extracted 3 sources.
All the three were merged into a single file and cleaned.The various data cleaning steps were
a)Removed duplicates lines
b)Removed lines containing html tags
c) Removed empty lines and lines which contains an empty english or tamil translation
d) Manually inspected and removed junk characters
There are total of 370,000 (roughly) lines of text which were split to train, validation and test sets. 
Script used for data cleaning :- https://colab.research.google.com/drive/1AViiS_4-ClngwZ4bKYbmlq3cVAnNHCk0#scrollTo=MWn-AuohbfXI

Note:-
For dataset-3, the data was created by non-expert translators hired over Mechanical Turk and the other 
datasets were created from extracting text from movie subtitles, Bible, Quran, cinema and news websites so the quality of translations are of mixed.
However, it should be useful enough to get you started training models.

Reference for dataset 1 :- http://ufal.mff.cuni.cz/~ramasamy/parallel/html/
Reference for dataset 2 :- http://opus.nlpl.eu/
Reference for dataset 3 :- https://github.com/joshua-decoder/indian-parallel-corpora
"""
test_file_link = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_test.tar.gz'
valid_file_link = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_valid.tar.gz'
train_file_link = 'https://drive.google.com/uc?export=download&id=1eet9TaAKoEXEU1gU4xGQxvhl6oJWAZbA'
#train_file_2_link = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_train_split2.tar.gz'
#train_file_3_link = 'https://github.com/praveenjune17/Neural-Machine-Translation-English-Tamil-model/raw/master/en_tam_parallel_text_train_split3.tar.gz'

class EnTamParallelText(tfds.core.GeneratorBasedBuilder):
  """(en_tam_parallel_text): English_Tamil parallel text corpus"""
  VERSION = tfds.core.Version('0.1.0')
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self, description=_DESCRIPTION, features=tfds.features.Translation(languages=("input", "target"), encoder_config=tfds.features.text.TextEncoderConfig()), urls=["http://ufal.mff.cuni.cz/~ramasamy/parallel/html/", "http://opus.nlpl.eu/", "https://github.com/joshua-decoder/indian-parallel-corpora"], \
        supervised_keys=("input", "target"), citation=_CITATION,)

  def _split_generators(self, dl_manager):
    """Load the data from the github links provided above"""
    extracted_path_test = dl_manager.download_and_extract(test_file_link)
    extracted_path_valid = dl_manager.download_and_extract(valid_file_link)
    extracted_path_train = dl_manager.download_and_extract(train_file_link)
    #extracted_path_train = dl_manager.extract([next(dl_manager.iter_archive(dl_manager.download(train_file_1_link)))[1], 
	#                            next(dl_manager.iter_archive(dl_manager.download(train_file_2_link)))[1],
	#                            next(dl_manager.iter_archive(dl_manager.download(train_file_3_link)))[1]])   
	#temp2 = dl_manager.iter_archive(train_file_2_link)
	#temp3 = dl_manager.iter_archive(train_file_3_link)
	#op_path = downloaded_path_train+
    #print('*********************')
    #print(tf.io.gfile.Glob(os.path.join(downloaded_path_train,'*.tar.gz')))
    #for file in [next(dl_manager.iter_archive(dl_manager.download(train_file_1_link)))[0],
    #             next(dl_manager.iter_archive(dl_manager.download(train_file_2_link)))[0],
    #             next(dl_manager.iter_archive(dl_manager.download(train_file_3_link)))[0]]:
    #  with (tf.io.gfile.GFile(file,'rb')) as fp:
    #    temp += fp.read()
    #    extracted_path_train = dl_manager.extract(temp)
    #  for filename in tf.io.gfile.Glob(os.path.join(downloaded_path_train,'*.tar.gz')):
    #    with open(filename,'rb') as readfile:
    #      shutil.copyfileobj(readfile, fp)
    #  extracted_path_train=dl_manager.extract(fp)
	  #tf.gfile.ListDirectory(extracted_path_train)
    #os.system('''cat /root/tensorflow_datasets/downloads/prav_Neur-Mach-Tran-Engl-Tami-mode_raw_a7-jw1afIzryZDxHLxn886ieInxXmgaP9rhhjeZwesI.tar.gz /root/tensorflow_datasets/downloads/prav_Neur-Mach-Tran-Engl-Tami-mode_raw___HgTPv00ZHCCeRb7VVkwQOvGJtMWRm8cSM9C2KYCP4.tar.gz /root/tensorflow_datasets/downloads/prav_Neur-Mach-Tran-Engl-Tami-mode_raw_7UHyQWdhPnGAORqS9kyetxl7rhl2a4n99boGVX65mfA.tar.gz > /root/tensorflow_datasets/downloads/en_tam_parallel_corpus_train.tar.gz''')
    #extracted_path_train = dl_manager.extract("../root/tensorflow_datasets/downloads/en_tam_parallel_corpus_train.tar.gz")
    return [
          tfds.core.SplitGenerator(name=tfds.Split.TRAIN, num_shards=2,\
              gen_kwargs={"data_file":os.path.join(extracted_path_train,'en_tam_parallel_corpus_train')}),
          tfds.core.SplitGenerator(name=tfds.Split.VALIDATION, num_shards=1,\
              gen_kwargs={"data_file":os.path.join(extracted_path_valid,'en_tam_parallel_corpus_validation')}),
          tfds.core.SplitGenerator(name=tfds.Split.TEST, num_shards=1,\
              gen_kwargs={"data_file":os.path.join(extracted_path_test,'en_tam_parallel_corpus_test')}),
      ]

  def _generate_examples(self, data_file):
    with tf.io.gfile.GFile(data_file) as dataset:
      for line in dataset:
        ip, target = line.strip().split('\t')
        yield {"input": ip, "target": target}
		