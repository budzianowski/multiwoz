# multiwoz

multiwoz is an open source toolkit for building end-to-end trainable task-oriented dialogue models.
It is released by Paweł Budzianowski from Cambridge Dialogue Systems Group under Apache License 2.0.

<h2>Benchmarks</h2>
<h3>Belief Tracking</h3>
<div class="datagrid" style="width:500px;">
<table>
<thead><tr><th>Model</th><th>Joint Accuracy</th><th>Slot</th>
</tr></thead>
<tbody>
<tr><td><a href="https://www.aclweb.org/anthology/P18-2069">MDBT</a> (Ramadan et al., 2018) </td><td>15.57 </td><td>89.53</td></tr>
<tr><td><a href="https://arxiv.org/abs/1805.09655">GLAD</a> (Zhong et al., 2018)</td><td>35.57</td><td>95.44 </td></tr>
<tr><td><a href="https://arxiv.org/pdf/1812.00899.pdf">GCE</a> (Nouri and Hosseini-Asl, 2018)</td><td>36.27</td><td>98.42</td></tr>
<tr><td><a href="https://arxiv.org/pdf/1905.08743.pdf">TRADE</a> (Wu et al, 2019)</td><td>48.62</td><td>96.92</td></tr>
</tbody>
</table>
</div>

<h3>Context-to-Text Generation</h3>
<div class="datagrid" style="width:500px;">
<table>
<thead><tr><th>Model</th><th>INFORM</th><th>SUCCESS</th><th>BLEU</th></tr></thead>
<tbody>
<tr><td><a href="https://pdfs.semanticscholar.org/47d0/1eb59cd37d16201fcae964bd1d2b49cfb55e.pdf">Baseline</a> (Budzianowski et al. 2018)</td><td>71.29</td><td> 60.96 </td><td> 18.8 </td></tr>
<tr><td><a href="https://arxiv.org/abs/1902.08858">LaRL</a> (Zhao et al. 2019)</td><td>82.78</td><td>79.2</td><td> 12.8</td></tr>
<tfoot> </tfoot>
</tbody>
</table>
</div>

<h3>Natural Language Generation</h3>
<div class="datagrid" style="width:500px;"><table>
<thead><tr><th>Model</th><th>SER</th><th>BLEU</th></tr></thead>
<tbody>
<tr><td><a href="https://pdfs.semanticscholar.org/47d0/1eb59cd37d16201fcae964bd1d2b49cfb55e.pdf">Baseline</a> (Budzianowski et al. 2018)</td><td>2.99 </td><td> 0.632</td></tr>
</tbody>
</table>
</div>

# Requirements
Python 2 with pip

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
To evaluate the run:

```python test.py [--args=value]```

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
```

# Bug Report

If you have found any bugs in the code, please contact: pfb30 at cam dot ac dot uk
