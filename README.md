# ResCNN_RelationExtraction
 Deep Residual Learning for Weakly-Supervised Relation Extraction: https://arxiv.org/abs/1707.08866
 By Yi Yao (Darren) Huang, [William Wang](https://www.cs.ucsb.edu/~william/)
 
### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Model](#model)
0. [Result](#result)

### Introduction
This work discuss about how we solve the noise from distant supervision. 
We propose the Deep Residual Learning for relation extraction and mitigate the influence from the noisy in semi-supervision training data.
This paper is published in EMNLP2017.

### Citation
If you use this model and the concept in your research, please cite:

      @InProceedings{huang-wang:2017:EMNLP2017,
          author    = {Huang, YiYao  and  Wang, William Yang},
          title     = {Deep Residual Learning for Weakly-Supervised Relation Extraction},
          booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
          month     = {September},
          year      = {2017},
          address   = {Copenhagen, Denmark},
          publisher = {Association for Computational Linguistics},
          pages     = {1804--1808},
          url       = {https://www.aclweb.org/anthology/D17-1191}
        }


### Mod@InProceedings{huang-wang:2017:EMNLP2017,
  author    = {Huang, YiYao  and  Wang, William Yang},
  title     = {Deep Residual Learning for Weakly-Supervised Relation Extraction},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {1804--1808},
  abstract  = {Deep residual learning (ResNet) is a new method for training very deep neural
	networks using identity mapping for shortcut connections. ResNet has won the
	ImageNet ILSVRC 2015 classification task, and achieved state-of-the-art
	performances in many computer vision tasks. However, the effect of residual
	learning on noisy natural language processing tasks is still not well
	understood. In this paper, we design a novel convolutional neural network (CNN)
	with residual learning, and investigate its impacts on the task of distantly
	supervised noisy relation extraction.  In contradictory to popular beliefs that
	ResNet only works well for very deep networks, we found  that even with 9
	layers of CNNs, using identity mapping could significantly improve the
	performance for distantly-supervised relation extraction.},
  url       = {https://www.aclweb.org/anthology/D17-1191}
}
el
![Architecture](https://user-images.githubusercontent.com/16465582/30602043-05f63dd6-9d96-11e7-9f2e-382e15a2b37a.png)


### Result
![Result](https://user-images.githubusercontent.com/16465582/30602544-6c3bd1a4-9d97-11e7-9f8f-807b436ede16.png)
