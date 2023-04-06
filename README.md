# SemEval2020-Task6

**Definition extraction** is an important task in natural language processing (NLP) as it can assist in
a wide range of applications such as question answering, information retrieval and knowledge base
construction. In this research, we focused on two of the three definition extraction tasks of the
SemEval 2020 [1] Task 6 [2], which aim to evaluate the performance of systems in extracting definitions
from text: **Sequence Labeling** and **Relation Extraction**.  

We approached the task of *sequence labeling* by using a combination of pre-trained BERT-based
models such as SciBERT and the traditional machine learning model CRF. We fine-tuned these
models on the task-specific dataset and evaluated them using standard metrics such as precision,
recall and F1-score. Our results show that our model achieves reasonable results with a F1-score
of 50.76%, as this would place us on the 26th place on the official leaderboard.

Furthermore, we also explored the use of data augmentation techniques such as oversampling and
synonym replacement to improve the performance of the models on the unbalanced dataset. Our
results showed that data augmentation techniques can improve the performance of the models by
more than 8 percentage points.

For the *relation extraction* task, we applied a feature-based method employing XGBoost classifier
with post-processing methods. This approach allowed us to reach high-quality predictions in terms
of macro-averaged F1-score over all 5 types of relations, which would place us at the 11-th place
of the official leaderboard.

---

In summary, this research highlights the importance of using pre-trained transformer-based models
and data augmentation techniques in definition extraction tasks. We also demonstrated the
effectiveness of these models in achieving reasonable performance on the sequence labeling task.
Moreover, we showed that in some relation extraction tasks, simple approaches involving shallow
learning models might compete with deep learning models.

### References 

[1] https://alt.qcri.org/semeval2020/  
[2] https://competitions.codalab.org/competitions/22759