# MultiWOZ
Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a fully-labeled collection of human-human written conversations spanning over multiple domains and topics. At a size of 10k dialogues, it is at least one order of magnitude larger than all previous annotated task-oriented corpora.


The newest, corrected version of the dataset is available at [MultiWOZ_2.2](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2) thanks to [the Google crew](https://arxiv.org/abs/2007.12720).

The new, corrected version of the dataset is available at [MultiWOZ_2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip) thanks to [the Amazon crew](https://arxiv.org/abs/1907.01669).

The dataset used in the EMNLP publication can be accessed at: [MultiWOZ_2.0](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.0.zip)

The dataset used in the ACL publication can be accessed at: [MultiWOZ_1.0](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_1.0.zip)

# Data structure
There are 3,406 single-domain dialogues that include booking if the domain allows for that and 7,032 multi-domain dialogues consisting of at least 2 up to 5 domains. To enforce reproducibility of results, the corpus was randomly split into a train, test and development set. The test and development sets contain 1k examples each. Even though all dialogues are coherent, some of them were not finished in terms of task description. Therefore, the validation and test sets only contain fully successful dialogues thus enabling a fair comparison of models. There are no dialogues from hospital and police domains in validation and testing sets.

Each dialogue consists of a goal, multiple user and system utterances as well as a belief state. Additionally, the task description in natural language presented to turkers working from the visitor’s side is added. Dialogues with MUL in the name refers to multi-domain dialogues. Dialogues with SNG refers to single-domain dialogues (but a booking sub-domain is possible). The booking might not have been possible to complete if fail_book option is not empty in goal specifications – turkers did not know about that.

The belief state have three sections: semi, book and booked. Semi refers to slots from a particular domain. Book refers to booking slots for a particular domain and booked is a sub-list of book dictionary with information about the booked entity (once the booking has been made). The goal sometimes was wrongly followed by the turkers which may results in the wrong belief state. The joint accuracy metrics includes ALL slots.

# FAQ
1. File names refer to two types of dialogues. The MUL and PMUL names refer to strictly multi domain dialogues (at least 2 main domains are involved) while the SNG, SSNG and WOZ names refer to single domain dialogues with potentially sub-domains like booking.
2. Only system utterances are annotated with dialogue acts – there are no annotations from the user side.
3. There is no 1-to-1 mapping between dialogue acts and sentences.
4. There is no dialogue state tracking labels for police and hospital as these domains are very simple. However, there are no dialogues with these domains in validation and testing sets either.
5. For the dialogue state tracking experiments please follow the datat processing and scoring scripts from the [TRADE](https://github.com/jasonwu0731/trade-dst) model (Wu et al. 2019).

<h2>Benchmarks</h2>
<h3>Belief Tracking</h3>
<div class="datagrid" style="width:500px;">
<table>
<thead><tr><th></th><th colspan="2">MultiWOZ 2.0</th><th colspan="2">MultiWOZ 2.1</th><th colspan="2">MultiWOZ 2.2</th></tr></thead>
<thead><tr><th>Model</th><th>Joint Accuracy</th><th>Slot</th><th>Joint Accuracy</th><th>Slot</th><th>Joint Accuracy</th><th>Slot</th></tr></thead>
<tbody>
<tr><td><a href="https://www.aclweb.org/anthology/P18-2069">MDBT</a> (Ramadan et al., 2018) </td><td>15.57 </td><td>89.53</td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.09655">GLAD</a> (Zhong et al., 2018)</td><td>35.57</td><td>95.44 </td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1812.00899.pdf">GCE</a> (Nouri and Hosseini-Asl, 2018)</td><td>36.27</td><td>98.42</td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1908.01946.pdf">Neural Reading</a> (Gao et al, 2019)</td><td>41.10</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1907.00883.pdf">HyST</a> (Goel et al, 2019)</td><td>44.24</td><td></td><td></td><td></td> <td></td><td></td></tr>
<tr><td><a href="https://www.aclweb.org/anthology/P19-1546/">SUMBT</a> (Lee et al, 2019)</td><td>46.65</td><td>96.44</td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1909.05855.pdf">SGD-baseline</a> (Rastogi et al, 2019)</td><td></td><td></td><td>43.4</td><td></td><td>42.0</td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1905.08743.pdf">TRADE</a> (Wu et al, 2019)</td><td>48.62</td><td>96.92</td><td>46.0</td><td></td><td>45.4</td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1909.00754.pdf">COMER</a> (Ren et al, 2019)</td><td>48.79</td><td></td><td></td><td></td<td></td><td></td><td></td></tr>
<tr><td><a href="https://www.aclweb.org/anthology/2020.acl-main.636.pdf">MERET</a> (Huang et al, 2020)</td><td>50.91</td><td>97.07</td><td></td><td></td<td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1911.06192.pdf">DSTQA</a> (Zhou et al, 2019)</td><td>51.44</td><td>97.24</td><td>51.17</td><td>97.21</td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.10447.pdf">SUMBT+LaRL</a> (Lee et al. 2020)</td><td>51.52</td><td>97.89</td><td> </td><td> </td><td> </td><td> </td></tr>
 <tr><td><a href="https://arxiv.org/pdf/1910.03544.pdf">DS-DST</a> (Zhang et al, 2019)</td><td></td><td></td><td>51.2</td><td></td><td>51.7</td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.08115.pdf">LABES-S2S</a> (Zhang et al, 2020)</td><td></td><td></td><td>51.45</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1910.03544.pdf">DST-Picklist</a> (Zhang et al, 2019)</td><td>54.39</td><td></td><td>53.3</td><td></td><td></td><td></td></tr>
 <tr><td><a href="https://arxiv.org/pdf/2009.12005.pdf">MinTL-BART</a> (Lin et al, 2020)</td><td>52.10</td><td></td><td>53.62</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ChenL.10030.pdf">SST</a> (Chen et al. 2020)</td><td></td><td></td><td>55.23</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/abs/2005.02877">TripPy</a> (Heck et al. 2020)</td><td></td><td></td><td>55.3</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2005.00796.pdf">SimpleTOD</a> (Hosseini-Asl et al. 2020)</td><td></td><td></td><td>56.45</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.13570.pdf">ConvBERT-DG + Multi</a> (Mehri et al. 2020)</td><td></td><td></td><td>58.7</td><td></td><td></td><td></td></tr>
 <tr><td><a href="https://openreview.net/forum?id=oyZxhRI2RiE">TripPy + SCoRe</a> (Yu et al. 2021)</td><td></td><td></td><td>60.48</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2010.12850.pdf">TripPy + CoCoAug</a> (Li and  Yavuz et al. 2020)</td><td></td><td></td><td>60.53</td><td></td><td></td><td></td></tr>
</tbody>
</table>
</div>



<h3>Policy Optimization</h3>
<div class="datagrid" style="width:500px;">
<table>
<thead><tr><th>(INFORM	+ SUCCESS)*0.5 +	BLEU</th><th colspan="3">MultiWOZ 2.0</th><th colspan="3">MultiWOZ 2.1</th></tr></thead>
<thead><tr><th>Model</th><th>INFORM</th><th>SUCCESS</th><th>BLEU</th><th>INFORM</th><th>SUCCESS</th><th>BLEU</th></tr></thead>
<tbody>
 <tr><td><a href="https://arxiv.org/pdf/1907.05346.pdf">TokenMoE*</a> (Pei et al. 2019)</td><td>75.30</td><td> 59.70</td><td> 16.81 </td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://pdfs.semanticscholar.org/47d0/1eb59cd37d16201fcae964bd1d2b49cfb55e.pdf">Baseline*</a> (Budzianowski et al. 2018)</td><td>71.29</td><td> 60.96 </td><td> 18.8 </td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/1907.10016.pdf">Structured Fusion*</a> (Mehri et al. 2019)</td><td>82.70</td><td>72.10</td><td> 16.34</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/abs/1902.08858">LaRL*</a> (Zhao et al. 2019)</td><td>82.8</td><td>79.2</td><td> 12.8</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2005.00796.pdf">SimpleTOD</a> (Hosseini-Asl et al. 2020)</td><td>88.9</td><td>67.1</td><td> 16.9</td><td> 85.1</td><td> 73.5</td><td> 16.22</td></tr>
<tr><td><a href="https://arxiv.org/pdf/1911.08151.pdf">MoGNet</a> (Pei et al. 2019)</td><td>85.3</td><td>73.30</td><td> 20.13</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/1905.12866.pdf">HDSA*</a> (Chen et al. 2019)</td><td>82.9</td><td>68.9</td><td> 23.6</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/abs/1910.03756">ARDM</a> (Wu et al. 2019)</td><td>87.4</td><td>72.8</td><td> 20.6</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/1911.10484.pdf">DAMD</a> (Zhang et al. 2019)</td><td>89.2</td><td>77.9</td><td> 18.6</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2005.05298.pdf">SOLOIST</a> (Peng et al. 2020)</td><td>89.60</td><td> 79.30</td><td> 18.3</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2004.12363.pdf">MarCo</a> (Wang et al. 2020)</td><td>92.30</td><td> 78.60</td><td> 20.02</td><td> 92.50</td><td> 77.80</td><td> 19.54</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2012.03539.pdf">UBAR</a> (Yang et al. 2020)</td><td>94.00</td><td> 83.60</td><td> 17.20</td><td> 92.70</td><td> 81.00</td><td> 16.70</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2006.06814.pdf">HDNO</a> (Wang et al. 2020)</td><td>96.40</td><td>84.70</td><td>18.85</td><td>92.80</td><td>83.00</td><td> 18.97</td></tr>
<tr><td><a href="https://www.aclweb.org/anthology/2020.coling-main.41.pdf">LAVA</a> (Lubis et al. 2020)</td><td>97.50</td><td>94.80</td><td>12.10</td><td>96.39</td><td>83.57</td><td>14.02</td></tr>

<tfoot> </tfoot>
</tbody>
</table>
</div>
* The results were obtained with a previous version of the evaluator. The performance on these works before the upgrade were underestimated. 

<h3>Natural Language Generation</h3>
<div class="datagrid" style="width:500px;"><table>
<thead><tr><th>Model</th><th>SER</th><th>BLEU</th></tr></thead>
<tbody>
<tr><td><a href="https://pdfs.semanticscholar.org/47d0/1eb59cd37d16201fcae964bd1d2b49cfb55e.pdf">Baseline</a> (Budzianowski et al. 2018)</td><td>2.99 </td><td> 0.632</td></tr>
</tbody>
</table>
</div>

<h3>End-to-End Modelling</h3>
<div class="datagrid" style="width:500px;">
<table>
<thead><tr><th>(INFORM	+ SUCCESS)*0.5 +	BLEU</th><th colspan="3">MultiWOZ 2.0</th><th colspan="3">MultiWOZ 2.1</th></tr></thead>
<thead><tr><th>Model</th><th>INFORM</th><th>SUCCESS</th><th>BLEU</th><th>INFORM</th><th>SUCCESS</th><th>BLEU</th></tr></thead>
<tbody>
<tr><td><a href="https://arxiv.org/pdf/1911.10484.pdf">DAMD</a> (Zhang et al. 2019)</td><td>76.3</td><td>60.4</td><td> 18.6</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.08115.pdf">LABES-S2S</a> (Zhang et al. 2020)</td><td></td><td></td><td> </td><td>78.07 </td><td> 67.06</td><td> 18.3</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2005.00796.pdf">SimpleTOD</a> (Hosseini-Asl et al. 2020)</td><td>84.4</td><td>70.1</td><td> 15.01</td><td> </td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2103.06648.pdf">DoTS</a> (Jeon et al. 2021)</td><td>86.59</td><td>74.14</td><td>15.06</td><td>86.65</td><td>74.18</td><td>15.90</td></tr>
 <tr><td><a href="https://arxiv.org/pdf/2005.05298.pdf">SOLOIST</a> (Peng et al. 2020)</td><td>85.50</td><td>72.90</td><td> 16.54</td><td> </td><td></td><td> </td></tr>
  <tr><td><a href="https://arxiv.org/pdf/2009.12005.pdf">MinTL-BART</a> (Lin et al. 2020)</td><td>84.88</td><td>74.91</td><td> 17.89</td><td> </td><td></td><td> </td></tr>
 <tr><td><a href="https://www.aclweb.org/anthology/2020.coling-main.41.pdf">LAVA</a> (Lubis et al. 2020)</td><td>91.80</td><td>81.80</td><td>12.03</td><td> </td><td> </td><td> </td></tr>
 <tr><td><a href="https://arxiv.org/pdf/2103.10518.pdf">NoisyChannel</a> (Liu et al. 2021)</td><td>86.90</td><td>76.20</td><td>20.58</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2012.03539.pdf">UBAR</a> (Yang et al. 2020)</td><td>95.40</td><td> 80.70</td><td> 17.00</td><td> 95.70</td><td> 81.80</td><td> 16.50</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.10447.pdf">SUMBT+LaRL</a> (Lee et al. 2020)</td><td>92.20</td><td> 85.40</td><td> 17.90</td><td> </td><td> </td><td> </td></tr>

<tfoot> </tfoot>
</tbody>
</table>
</div>


# Requirements
Python 2 with pip, pytorch==0.4.1

# Quick start
In repo directory:

## Preprocessing
To download and pre-process the data run:

```python create_delex_data.py```

## Training
To train the model run:

```python train.py [--args=value]```

Some of these args include:

```
// hyperparamters for model learning
--max_epochs        : numbers of epochs
--batch_size        : numbers of turns per batch
--lr_rate           : initial learning rate
--clip              : size of clipping
--l2_norm           : l2-regularization weight
--dropout           : dropout rate
--optim             : optimization method

// network structure
--emb_size          : word vectors emedding size
--use_attn          : whether to use attention
--hid_size_enc      : size of RNN hidden cell
--hid_size_pol      : size of policy hidden output
--hid_size_dec      : size of RNN hidden cell
--cell_type         : specify RNN type
```

## Testing
To evaluate the trained model, run:

```python test.py [--args=value]```

To evaluate the outside model, run:

```python evaluate.py```

where in line 611 you need to load your generation predictions.


# Benchmark results
The following [benchmark results](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/) were produced by this software.
We ran a small grid search over various hyperparameter settings
and reported the performance of the best model on the test set.
The selection criterion was 0.5*match + 0.5*success+100*BLEU on the validation set.
The final parameters were:

```
// hyperparamters for model learning
--max_epochs        : 20
--batch_size        : 64
--lr_rate           : 0.005
--clip              : 5.0
--l2_norm           : 0.00001
--dropout           : 0.0
--optim             : Adam

// network structure
--emb_size          : 50
--use_attn          : True
--hid_size_enc      : 150
--hid_size_pol      : 150
--hid_size_dec      : 150
--cell_type         : lstm
```


# References
If you use any source codes or datasets included in this toolkit in your
work, please cite the corresponding papers. The bibtex are listed below:
```
[Budzianowski et al. 2018]
@inproceedings{budzianowski2018large,
    Author = {Budzianowski, Pawe{\l} and Wen, Tsung-Hsien and Tseng, Bo-Hsiang  and Casanueva, I{\~n}igo and Ultes Stefan and Ramadan Osman and Ga{\v{s}}i\'c, Milica},
    title={MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2018}
}

[Ramadan et al. 2018]
@inproceedings{ramadan2018large,
  title={Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing},
  author={Ramadan, Osman and Budzianowski, Pawe{\l} and Gasic, Milica},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  volume={2},
  pages={432--437},
  year={2018}
}

[Eric et al. 2019]
@article{eric2019multiwoz,
  title={MultiWOZ 2.1: Multi-Domain Dialogue State Corrections and State Tracking Baselines},
  author={Eric, Mihail and Goel, Rahul and Paul, Shachi and Sethi, Abhishek and Agarwal, Sanchit and Gao, Shuyag and Hakkani-Tur, Dilek},
  journal={arXiv preprint arXiv:1907.01669},
  year={2019}
}

[Zang et al. 2020]
@inproceedings{zang2020multiwoz,
  title={MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines},
  author={Zang, Xiaoxue and Rastogi, Abhinav and Sunkara, Srinivas and Gupta, Raghav and Zhang, Jianguo and Chen, Jindong},
  booktitle={Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, ACL 2020},
  pages={109--117},
  year={2020}
}
```

# License
MultiWOZ is an open source toolkit for building end-to-end trainable task-oriented dialogue models.
It is released by Paweł Budzianowski from Cambridge Dialogue Systems Group under Apache License 2.0.

# Bug Report
If you have found any bugs in the code, please contact: budzianowski at gmail dot com
