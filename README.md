# Nugget Proposal Networks for Chinese Event Detection

This is the source code for paper "Nugget Proposal Networks for Chinese Event Detection" in ACL2018.

## Requirements

* Tensorflow >= 1.2.0

## Usage
First, please unzip the word2vec embeddings in "src/"

* gzip -d src/word_word2vec.dat.gz
* gzip -d src/char_word2vec.dat.gz

Then enter src dir, run the program like

* python NPN_TS.py kbp_config.cfg

Hyperparameters in our paper are saved in configure file "kbp_config.cfg" or "ace_config.cfg".

## Citation
Please cite:
* Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun. *Nugget Proposal Networks for Chinese Event Detection*. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics.

```
@InProceedings{lin-Etal:2018:ACL2018nugget,
  author    = {Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le},
  title     = {Nugget Proposal Networks for Chinese Event Detection},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  month     = {July},
  year      = {2018},
  location   = {Melbourne, Australia},
  publisher = {Association for Computational Linguistics},
}
```

## Contact
If you have any question or want to request for the data(only if you have the license from LDC), please contact me by
* hongyu2016@iscas.ac.cn
