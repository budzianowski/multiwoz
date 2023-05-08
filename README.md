# MultiWOZ
Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a fully-labeled collection of human-human written conversations spanning over multiple domains and topics. At a size of 10k dialogues, it is at least one order of magnitude larger than all previous annotated task-oriented corpora.

## Versions

 - **The newest, corrected version of the dataset is available at [MultiWOZ_2.2](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2) thanks to [the Google crew](https://arxiv.org/abs/2007.12720).**
 - The new, corrected version of the dataset is available at [MultiWOZ_2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip) thanks to [the Amazon crew](https://arxiv.org/abs/1907.01669).
 - The dataset used in the EMNLP publication can be accessed at: [MultiWOZ_2.0](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.0.zip)
 - The dataset used in the ACL publication can be accessed at: [MultiWOZ_1.0](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_1.0.zip)

## Data structure
There are 3,406 single-domain dialogues that include booking if the domain allows for that and 7,032 multi-domain dialogues consisting of at least 2 up to 5 domains. To enforce reproducibility of results, the corpus was randomly split into a train, test and development set. The test and development sets contain 1k examples each. Even though all dialogues are coherent, some of them were not finished in terms of task description. Therefore, the validation and test sets only contain fully successful dialogues thus enabling a fair comparison of models. There are no dialogues from hospital and police domains in validation and testing sets.

Each dialogue consists of a goal, multiple user and system utterances as well as a belief state. Additionally, the task description in natural language presented to turkers working from the visitor’s side is added. Dialogues with MUL in the name refers to multi-domain dialogues. Dialogues with SNG refers to single-domain dialogues (but a booking sub-domain is possible). The booking might not have been possible to complete if fail_book option is not empty in goal specifications – turkers did not know about that.

The belief state have three sections: semi, book and booked. Semi refers to slots from a particular domain. Book refers to booking slots for a particular domain and booked is a sub-list of book dictionary with information about the booked entity (once the booking has been made). The goal sometimes was wrongly followed by the turkers which may results in the wrong belief state. The joint accuracy metrics includes ALL slots.

# :grey_question: FAQ
- File names refer to two types of dialogues. The `MUL` and `PMUL` names refer to strictly multi domain dialogues (at least 2 main domains are involved) while the `SNG`, `SSNG` and `WOZ` names refer to single domain dialogues with potentially sub-domains like booking.
- Only system utterances are manually annotated with dialogue acts – there are no human annotations from the user side. But MultiWOZ 2.1 automatically annotated user dialogue acts via heuristics developed in [ConvLab](https://arxiv.org/abs/1904.08637).
- There is no 1-to-1 mapping between dialogue acts and sentences.
- There is no dialogue state tracking labels for police and hospital as these domains are very simple. However, there are no dialogues with these domains in validation and testing sets either.

# :trophy: Benchmarks
## Dialog State Tracking

:bangbang: **For the DST experiments please follow the data processing and scoring scripts from the [TRADE model](https://github.com/jasonwu0731/trade-dst).**

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
<tr><td><a href="https://arxiv.org/pdf/2203.08568.pdf">In-Context Learning (Codex)</a> (Hu et al. 2022)</td><td></td><td></td><td>50.65<td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1911.06192.pdf">DSTQA</a> (Zhou et al, 2019)</td><td>51.44</td><td>97.24</td><td>51.17</td><td>97.21</td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.10447.pdf">SUMBT+LaRL</a> (Lee et al. 2020)</td><td>51.52</td><td>97.89</td><td> </td><td> </td><td> </td><td> </td></tr>
 <tr><td><a href="https://arxiv.org/pdf/1910.03544.pdf">DS-DST</a> (Zhang et al, 2019)</td><td></td><td></td><td>51.2</td><td></td><td>51.7</td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.08115.pdf">LABES-S2S</a> (Zhang et al, 2020)</td><td></td><td></td><td>51.45</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/1910.03544.pdf">DST-Picklist</a> (Zhang et al, 2019)</td><td>54.39</td><td></td><td>53.3</td><td></td><td></td><td></td></tr>
 <tr><td><a href="https://arxiv.org/pdf/2009.12005.pdf">MinTL-BART</a> (Lin et al, 2020)</td><td>52.10</td><td></td><td>53.62</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ChenL.10030.pdf">SST</a> (Chen et al. 2020)</td><td></td><td></td><td>55.23</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/abs/2005.02877">TripPy</a> (Heck et al. 2020)</td><td></td><td></td><td>55.3</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2005.00796.pdf">SimpleTOD</a> (Hosseini-Asl et al. 2020)</td><td></td><td></td><td>56.45</td><td></td><td></td><td></td></tr>
 <tr><td><a href="https://arxiv.org/pdf/2109.14739.pdf">PPTOD</a> (Su et al. 2021)</td><td>53.89</td><td></td><td>57.45</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.13570.pdf">ConvBERT-DG + Multi</a> (Mehri et al. 2020)</td><td></td><td></td><td>58.7</td><td></td><td></td><td></td></tr>
 <tr><td><a href="https://arxiv.org/abs/2112.08321">PrefineDST</a> (Cho et al. 2021)</td><td></td><td></td><td>58.9* (53.8)</td><td></td><td></td><td></td></tr>
 <tr><td><a href="https://openreview.net/forum?id=oyZxhRI2RiE">TripPy + SCoRe</a> (Yu et al. 2021)</td><td></td><td></td><td>60.48</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2010.12850.pdf">TripPy + CoCoAug</a> (Li and  Yavuz et al. 2020)</td><td></td><td></td><td>60.53</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/abs/2106.00291">TripPy + SaCLog</a> (Dai et al. 2021)</td><td></td><td></td><td>60.61</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://aclanthology.org/2021.emnlp-main.620.pdf">KAGE-GPT2</a> (Lin et al, 2021)</td><td>54.86</td><td>97.47</td><td></td><td></td><td></td><td></td></tr>
<tr><td><a href="https://aclanthology.org/2021.nlp4convai-1.8/">AG-DST</a> (Tian et al. 2021)</td><td></td><td></td><td></td><td></td><td>57.26</td><td></td></tr>
<tr><td><a href="https://aclanthology.org/2021.emnlp-main.404.pdf">SDP-DST</a> (Lee et al. 2021)</td><td></td><td></td><td>56.66<td></td><td>57.60</td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2110.11205v3.pdf">DAIR</a> (Huang et al. 2022)</td><td></td><td></td><td></td><td></td><td>59.98</td><td></td></tr>
</tbody>
</table>

Note: *SimpleTOD's evaluation setting does not distinguish between `dontcare` and `none` slot values, which results in an inflated JGA. Results when this discrepancy is resolved are shown in parantheses. Refer more details to the CheckDST github for a corrected evaluation script: https://github.com/wise-east/checkdst.


</div>

## Response Generation

:bangbang: **For the response generation evaluation please see and use the scoring scripts from [this repository](https://github.com/Tomiinek/MultiWOZ_Evaluation).** 

- See [this directory](https://github.com/Tomiinek/MultiWOZ_Evaluation/tree/master/predictions) for details about the raw generated predictions of other models.
- Inform meaures whether the system provides an appropriate entity and Success measures whether the system answers all the requested attributes.
- BLEU reported in these tables is calculated with references obtained from the *MultiWOZ 2.2 span annotations*.
- CBE stands for *conditional bigram entropy*. 

| Model              | BLEU | Inform  | Success  | Av. len. | CBE | #uniq. words | #uniq. 3-grams |
| ------------------ | -----:| -------:| --------:| ---------:| -----------------:| -------------:| -------------:| 
| Reference corpus &nbsp; | -    | 93.7 | 90.9 | 14.00 | 3.01 | 1407 | 23877 | 

**End-to-end models**, i.e. those that use only the dialogue context as input to generate responses. 
##### Combined Score = (INFORM	+ SUCCESS)*0.5 + BLEU

| Model              | BLEU | Inform  | Success  | Combined Score |Av. len. | CBE | #uniq. words | #uniq. 3-grams |
| ------------------ | :-----:| :-------:| :--------:| :---------:|:---------:| :-----------------:| :-------------:| :-------------:| 
| LABES ([paper](https://arxiv.org/pdf/2009.08115v3.pdf)\|[code](https://github.com/thu-spmi/LABES)) | 18.9 | 68.5 | 58.1 | 82.2|14.20 | 1.83 | 374  | 3228  |
| DAMD ([paper](https://arxiv.org/abs/1911.10484)\|[code](https://github.com/thu-spmi/damd-multiwoz))  | 16.4 | 57.9 | 47.6 | 84.8|14.27 | 1.65 | 212  | 1755  |
| AuGPT ([paper](https://arxiv.org/abs/2102.05126)\|[code](https://github.com/ufal/augpt)) | 16.8 | 76.6 | 60.5 |85.4 |12.90 | 2.15 | 608  | 5843  |
| MinTL ([paper](https://arxiv.org/pdf/2009.12005.pdf)\|[code](https://github.com/zlinao/MinTL)) | 19.4 | 73.7 | 65.4 | 89.0|14.78 | 1.81 | 297  | 2525  |
| SOLOIST ([paper](https://arxiv.org/abs/2005.05298))  | 13.6 | 82.3 | 72.4 | 90.9|18.45 | **2.41** | **615**  | **7923**  |
| DoTS ([paper](https://arxiv.org/pdf/2103.06648.pdf))  | 16.8 | 80.4 | 68.7 | 91.4|14.66 | 2.10 | 411  | 5162  |
| UBAR ([paper](https://arxiv.org/abs/2012.03539)\|[code](https://github.com/TonyNemo/UBAR-MultiWOZ))  | 17.6 | 83.4 | 70.3 | 94.4|13.54 | 2.10 | 478  | 5238  |
| PPTOD ([paper](https://arxiv.org/abs/2109.14739)\|[code](https://github.com/awslabs/pptod))  | 18.2 | 83.1 | 72.7 | 96.1|12.73 | 1.88 | 301  | 2538  |
| RSTOD ([paper](https://arxiv.org/abs/2208.07097)\|[code](https://github.com/radi-cho/rstod))  | 18.0 | 83.5 | 75.0 | 97.3 | 13.64 | 1.84 | 376  | 3162  |
| BORT ([paper](https://arxiv.org/abs/2205.02471)\|[code](https://github.com/JD-AI-Research-NLP/BORT))  | 17.9 | 85.5 | 77.4 | 99.4|14.91 | 1.88 | 294  | 2492  |
| MTTOD ([paper](https://aclanthology.org/2021.findings-emnlp.112.pdf)\|[code](https://github.com/bepoetree/MTTOD))  | 19.0 | 85.9 | 76.5 | 100.2 |13.94 | 1.93 | 514  | 4066  |
| GALAXY ([paper](https://arxiv.org/abs/2111.14592)\|[code](https://github.com/siat-nlp/GALAXY)) |19.64| 85.4 | 75.7 |100.2| 13.39 | 1.75 | 295 | 2275 |
| RewardNet([paper](https://arxiv.org/pdf/2302.10342.pdf)\|[code](https://github.com/Shentao-YANG/Fantastic_Reward_ICLR2023))| 17.6 | 87.6 | **81.5** | 102.2 | 13.22 | 1.99 | 423  | 3942  |
| Mars ([paper](https://arxiv.org/abs/2210.08917))  | **19.9** | 88.9 | 78.0 | 103.4 |13.93 | 1.65 | 288  | 2264  |
| KRLS ([paper](https://arxiv.org/pdf/2211.16773))  | 19.0 | **89.2** | 80.3 | **103.8** | 13.79 | 1.90 | 494  | 3884  |



**Policy optimization models**, i.e. those that use also the ground-truth dialog states to generate responses.
##### Combined Score = (INFORM	+ SUCCESS)*0.5 + BLEU

| Model              | BLEU | Inform  | Success  | Combined Score|Av. len. | CBE | #uniq. words | #uniq. 3-grams |
| ------------------ | :-----:| :-------:| :--------:| :---------:|:---------:| :-----------------:| :-------------:| :-------------:|
| UniConv ([paper](https://arxiv.org/pdf/2004.14307.pdf)\|[code](https://github.com/henryhungle/UniConv)) | 18.1 | 66.7 | 58.7 | 80.8|14.17 | 1.79 | **338** | 2932 |
| SFN ([paper](https://arxiv.org/pdf/1907.10016.pdf)\|[code](https://github.com/Shikib/structured_fusion_networks))     | 14.1 | 93.4 | 82.3 | 101.9|14.93 | 1.63 | 188 | 1218 |
| HDSA ([paper](https://arxiv.org/pdf/1905.12866.pdf)\|[code](https://github.com/wenhuchen/HDSA-Dialog))    | **20.7** | 87.9 | 79.4 | 104.4|14.42 | 1.64 | 259 | 2019 |
| LAVA ([paper](https://arxiv.org/abs/2011.09378)\|[code](https://gitlab.cs.uni-duesseldorf.de/general/dsml/lava-public/-/tree/master/experiments_woz/sys_config_log_model/2020-05-12-14-51-49-actz_cat))    | 10.8 | **95.9** | **93.5** | 105.5|13.28 | 1.27 | 176 | 708  |
| HDNO ([paper](https://arxiv.org/pdf/2006.06814.pdf)\|[code](https://github.com/mikezhang95/HDNO))    | 17.8 | 93.3 | 83.4 | 106.1|14.96 | 0.84 | 103 | 315  |
| KRLS ([paper](https://arxiv.org/pdf/2211.16773))  | 19.1 | 93.1 | 83.7 | 107.5 | 13.82 | 1.90 | 489  | 3885  |
| MarCo ([paper](https://arxiv.org/pdf/2004.12363.pdf)\|[code](https://github.com/InitialBug/MarCo-Dialog))   | 17.3 | 94.5 | 87.2 | **108.1**|16.01 | **1.94** | 319 | **3002** |
| GALAXY ([paper](https://arxiv.org/abs/2111.14592)\|[code](https://github.com/siat-nlp/GALAXY)) | 19.92 | 92.8 | 83.5 |**108.1**| 13.68 | 1.75 | 281 | 2344 |

### Older results

The following tables show older numbers which may not be comparable directly because of inconsistencies in the evaluation scripts used.

\* Denotes that the results were obtained with an even earlier version of the evaluator. The performance on these works were underestimated. 

**End-to-end models**

<div class="datagrid" style="width:500px;">
<table>
<thead><tr><th>(INFORM	+ SUCCESS)*0.5 +	BLEU</th><th colspan="3">MultiWOZ 2.0</th><th colspan="3">MultiWOZ 2.1</th></tr></thead>
<thead><tr><th>Model</th><th>INFORM</th><th>SUCCESS</th><th>BLEU</th><th>INFORM</th><th>SUCCESS</th><th>BLEU</th></tr></thead>
<tbody>
<tr><td><a href="https://arxiv.org/pdf/1911.10484.pdf">DAMD</a> (Zhang et al. 2019)</td><td>76.3</td><td>60.4</td><td> 16.6</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.08115.pdf">LABES-S2S</a> (Zhang et al. 2020)</td><td></td><td></td><td> </td><td>78.07 </td><td> 67.06</td><td> 18.3</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2005.00796.pdf">SimpleTOD</a> (Hosseini-Asl et al. 2020)</td><td>84.4</td><td>70.1</td><td> 15.01</td><td> </td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2103.06648.pdf">DoTS</a> (Jeon et al. 2021)</td><td>86.59</td><td>74.14</td><td>15.06</td><td>86.65</td><td>74.18</td><td>15.90</td></tr>
<tr><td><a href="https://aclanthology.org/2021.acl-long.13.pdf">JOUST</a> (Tseng et al. 2021)</td><td>83.20</td><td>73.50</td><td>17.60</td><td> </td><td> </td><td> </td></tr>
 <tr><td><a href="https://arxiv.org/pdf/2005.05298.pdf">SOLOIST</a> (Peng et al. 2020)</td><td>85.50</td><td>72.90</td><td> 16.54</td><td> </td><td></td><td> </td></tr>
  <tr><td><a href="https://arxiv.org/pdf/2009.12005.pdf">MinTL-BART</a> (Lin et al. 2020)</td><td>84.88</td><td>74.91</td><td> 17.89</td><td> </td><td></td><td> </td></tr>
 <tr><td><a href="https://www.aclweb.org/anthology/2020.coling-main.41.pdf">LAVA</a> (Lubis et al. 2020)</td><td>91.80</td><td>81.80</td><td>12.03</td><td> </td><td> </td><td> </td></tr>
 <tr><td><a href="https://arxiv.org/pdf/2103.10518.pdf">NoisyChannel</a> (Liu et al. 2021)</td><td>86.90</td><td>76.20</td><td>20.58</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2109.14739.pdf">PPTOD</a> (Su et al. 2021)</td><td>89.20</td><td>79.40</td><td>18.62</td><td>87.09</td><td>79.08</td><td> 19.17</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2012.03539.pdf">UBAR</a> (Yang et al. 2020)</td><td>95.40</td><td> 80.70</td><td> 17.00</td><td> 95.70</td><td> 81.80</td><td> 16.50</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2107.03286.pdf">DORA</a> (Jeon et al. 2021)</td><td>94.60</td><td>92.00</td><td>12.70</td><td>94.40</td><td>91.10</td><td>12.58</td></tr>
<tr><td><a href="https://arxiv.org/pdf/2009.10447.pdf">SUMBT+LaRL</a> (Lee et al. 2020)</td><td>92.20</td><td> 85.40</td><td> 17.90</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://arxiv.org/pdf/2103.06370.pdf">CASPI</a> (Ramachandran et al. 2021)</td><td>94.59</td><td>85.59</td><td>17.96</td><td> </td><td> </td><td> </td></tr>
<tr><td><a href="https://aclanthology.org/2021.findings-emnlp.112.pdf">MTTOD</a> (Lee 2021)</td><td>90.99</td><td> 82.58</td><td> 20.25</td><td> 90.99</td><td> 82.08</td><td> 19.68</td></tr>
<tr><td><a href="https://arxiv.org/abs/2205.02471">BORT</a> (Sun et al. 2022)</td><td>93.80</td><td>85.80</td><td>18.50</td><td> </td><td> </td><td></td></tr>
<tr><td><a href="https://arxiv.org/abs/2111.14592">GALAXY</a> (He et al. 2021)</td><td>94.40</td><td> 85.30</td><td> 20.50</td><td> 95.30</td><td> 86.20</td><td> 20.01</td></tr>
 <tfoot> </tfoot>
</tbody>
</table>
</div>


**Policy optimization models**

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
<tr><td><a href="https://aclanthology.org/2021.acl-long.13.pdf">JOUST</a> (Tseng et al. 2021)</td><td>94.70</td><td>86.70</td><td>18.70</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/pdf/2103.06370.pdf">CASPI</a> (Ramachandran et al. 2021)</td><td>96.80</td><td>87.30</td><td>19.10</td><td></td><td></td><td></td></tr>
<tr><td><a href="https://arxiv.org/abs/2111.14592">GALAXY</a> (He et al. 2021)</td><td>94.8</td><td>85.7</td><td>19.93</td><td>94.8</td><td>86.2</td><td>20.29</td></tr>
<tfoot> </tfoot>
</tbody>
</table>
</div>

# :thought_balloon: References
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

# Baseline

:bangbang: This part relates to the first version of the dataset and evaluation scripts.

### Requirements
Python 2 with `pip`, `pytorch==0.4.1`

### Preprocessing
To download and pre-process the data run:

```python create_delex_data.py```

### Training
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

## Results
We ran a small grid search over various hyperparameter settings and reported the performance of the best model on the test set.
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

# License
MultiWOZ is an open source toolkit for building end-to-end trainable task-oriented dialogue models.
It is released by Paweł Budzianowski from Cambridge Dialogue Systems Group under the MIT License.

# Bug Report
If you have found any bugs in the code, please contact: budzianowski@gmail.com or jianguozhang@salesforce.com
