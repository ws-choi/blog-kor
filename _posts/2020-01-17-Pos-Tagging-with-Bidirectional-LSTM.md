---
title: "Bidirectional RNN과 Bidirectional LSTM (실습편)"
layout: post
math: true
date: 2019-01-17
categories: NLP DeepLearning
permalink: nlp/deeplearning/Pos-Tagging-with-Bidirectional-LSTM/
---

이번 포스트에서는 Bidirectional LSTM Network를 이용하여 Part-of-Speech Tagging (PoS Tagging)을 실습해본다.
본 실습 예제는 [PyTorch 튜토리얼](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)을 참고하여 작성하였다.


## Preliminaries



- 본 포스트의 실습은 이전 포스트와 달리 `numpy`가 아닌 `PyTorch`를 이용함!
- 본 포스트에서는 `Natural Language Toolkit` 라이브러리를 사용함!

### Natural Language Toolkit 설치 및 데이터 셋 받기

Natural Language Toolkit (NLTK)가 깔려있지 않다면

```
conda install nltk
```

와 같은 방식으로 nltk를 깔아주자.


nltk 라이브러리를 설치했더라도, 데이터 셋을 다운받으려면 다음과 같은 추가적인 과정이 필요하다.

```shell
import nltk
nltk.download('universal_tagset()')
```

본 예제에서는 `brown corpus`의 `universal_tagset` 를 사용할 것이므로 위와 같이 받아주자.

### 데이터 살펴보기

다음과 같이 `brown corpus`의 `universal_tagset` 를 받아와서 변수로 만들어보자.


```python
from nltk.corpus import brown
brown_news_tagged = brown.tagged_sents(tagset='universal')

import numpy as np

print('The number of sentences in this corpus: %d \n' % len(brown_news_tagged))
print('Sample sentance: ')
print(np.array(brown_news_tagged[0]).T)
```

    The number of sentences in this corpus: 57340

    Sample sentance:
    [['The' 'Fulton' 'County' 'Grand' 'Jury' 'said' 'Friday' 'an'
      'investigation' 'of' "Atlanta's" 'recent' 'primary' 'election'
      'produced' '``' 'no' 'evidence' "''" 'that' 'any' 'irregularities'
      'took' 'place' '.']
     ['DET' 'NOUN' 'NOUN' 'ADJ' 'NOUN' 'VERB' 'NOUN' 'DET' 'NOUN' 'ADP'
      'NOUN' 'ADJ' 'NOUN' 'NOUN' 'VERB' '.' 'DET' 'NOUN' '.' 'ADP' 'DET'
      'NOUN' 'VERB' 'NOUN' '.']]


### 단어 사전 만들기!

인풋 데이터 집합을 만들어주기 위한 사전작업으로, 단어 사전을 만들어보자. [Vanilla RNN 실습]({{site.baseurl}}/nlp/deeplearning/Vanilla-RNN-%EC%8B%A4%EC%8A%B5/)때와 유사한 방법을 사용할 것이다.


```python
def extract_sentence (pair):
    return [s for (s, t) in pair]

sentences = [ extract_sentence(pair) for pair in brown_news_tagged ]
flatten = [word for sentence in sentences for word in sentence ]
unique_words = list(set(flatten))
word_to_idx = { word : i for i,word in enumerate(unique_words) }
idx_to_word = { i:word  for i,word in enumerate(unique_words) }
vocab_size = len(unique_words)

print('There are %d unique words in the data set.' % vocab_size)
```

    There are 56057 unique words in the data set.



```python
import torch

def word2id (sentence):
    return torch.tensor([word_to_idx[word] for word  in sentence ], dtype=torch.long)

word2id(['Hello', 'my','name','is', 'john', 'hello', 'Hello'])
```




    tensor([48582, 36790, 35040, 35000, 12817, 41889, 48582])



### Tag 사전 만들기!

정답 데이터 집합을 만들어주기 위한 사전작업으로, Tag 사전을 만들어보자.


```python
def extract_tag (pair):
    return [t for (s, t) in pair]

sentence_tags = [ extract_tag (pair) for pair in brown_news_tagged ]
flatten = [tag for sentence_tag in sentence_tags for tag in sentence_tag ]
unique_tags = list(set(flatten))
tag_to_idx = { word : i for i,word in enumerate(unique_tags) }
tag_size = len(unique_tags)

print('There are %d unique tags in the data set.' % tag_size)
print(tag_to_idx)
```

    There are 12 unique tags in the data set.
    {'ADP': 0, 'X': 1, 'VERB': 2, '.': 3, 'ADJ': 4, 'ADV': 5, 'PRT': 6, 'NOUN': 7, 'DET': 8, 'PRON': 9, 'NUM': 10, 'CONJ': 11}



```python
def tag2id (tags):
    return torch.tensor([tag_to_idx[tag] for tag  in tags ], dtype=torch.long)

tag2id(['ADJ', 'ADP','ADV','CONJ', 'CONJ', 'PRT', 'X'])
```




    tensor([ 4,  0,  5, 11, 11,  6,  1])



### Training Data 집합과 Test Data 집합 만들기

전체 데이터의 75%는 트레이닝 데이터로, 25%는 테스트 데이터로 만들어주자!


```python
import random

shuffled = list(brown_news_tagged)
random.seed(1)
random.shuffle(shuffled)

split_idx = (int)(0.75 * len(brown_news_tagged) )

training_data = shuffled[:split_idx]
test_data = shuffled[split_idx:]
```

## PyTorch word embedding 사용해보기


```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

embeds = nn.Embedding(vocab_size, 5)  # (vocab_size, embedding_size)
lookup_tensor = torch.tensor([word_to_idx["words"]], dtype=torch.long)
words_embed = embeds(lookup_tensor)
print(words_embed)
```

    tensor([[ 0.1171,  0.0513,  0.8599, -0.8551, -0.3351]],
           grad_fn=<EmbeddingBackward>)



```python
lstm = nn.LSTM(embedding_dim, 2)
first_hidden = (torch.zeros(1, 1, 2) ,
                torch.zeros(1, 1, 2))

sent = prepared_training_data[0]
length = len(sent)
embeded_sent = embeds(sentence)

lstm( embeded_sent.view(length, 1, -1), first_hidden )
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-857c8f313cb1> in <module>
    ----> 1 lstm = nn.LSTM(embedding_dim, 2)
          2 first_hidden = (torch.zeros(1, 1, 2) ,
          3                 torch.zeros(1, 1, 2))
          4
          5 sent = prepared_training_data[0]


    NameError: name 'embedding_dim' is not defined



```python
sentence = sentences[0][:5]
print('The original sentence: %s \n' % sentence)
word_ids = word2id(sentence)
print('Ids: %s: \n' % word_ids)
sentence_embed = embeds(word_ids)
print('sentence_embed\'s shape:\n%s:' % str(sentence_embed))
```

    The original sentence: ['The', 'Fulton', 'County', 'Grand', 'Jury']

    Ids: tensor([23217, 54967, 31112, 23732, 24743]):

    sentence_embed's shape:
    tensor([[-0.6171, -0.1046, -1.2225,  2.1271,  1.8479],
            [-0.5408, -0.0800, -0.3077, -0.6716,  0.1900],
            [-0.1758, -0.3613, -0.2091,  0.0833, -0.6361],
            [-1.3710,  1.6590, -0.2266,  0.4978,  0.1015],
            [-1.0120, -0.0862, -2.0142, -2.0842,  0.6107]],
           grad_fn=<EmbeddingBackward>):


## 모델 만들기

Pos Tagging을 위해 다음과 같은 모델을 만들것이다.

![Imgur](https://i.imgur.com/F79G5dW.png)


```python
from torch.nn.parameter import Parameter
from torch.nn.modules   import Module
from torch.nn import init
import math
```


```python
class BD_LSTM_Tagger(Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BD_LSTM_Tagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_LtoR = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm_RtoL = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2y = nn.Linear(hidden_dim*2, tag_size)
        self.hidden_LtoR = self.init_hidden()
        self.hidden_RtoL = self.init_hidden()

    def init_hiddens(self):
            self.hidden_LtoR = self.init_hidden()
            self.hidden_RtoL = self.init_hidden()


    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim) ,
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):

        length = len(sentence)

        embeded_sent = self.word_embeddings(sentence)
        lstm_out_LtoR, self.hidden_LtoR = self.lstm_LtoR( embeded_sent.view(length, 1, -1), self.hidden_LtoR)

        reversed_embeds = embeded_sent # reversed(self.embeded_sent)
        lstm_out_RtoL, self.hidden_RtoL = self.lstm_RtoL( reversed_embeds.view(length, 1, -1), self.hidden_RtoL)

        hidden_full = torch.cat((lstm_out_LtoR, lstm_out_RtoL), 2)

        y = self.hidden2y(hidden_full).view(length,-1)
        return F.log_softmax(y, dim=1)

```

## 훈련시키기


```python
def prepare_xy (tagged_sentence):
    sentence = extract_sentence(tagged_sentence)
    tags     = extract_tag(tagged_sentence)
    return word2id(sentence), tag2id(tags)    


embedding_dim = 100
hidden_dim = 200
model = BD_LSTM_Tagger(embedding_dim, hidden_dim, vocab_size, tag_size)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# sentence, tags = prepare_xy(training_data[0])

```


```python
def get_accuracy(prepared_data):
    correct_case = 0
    total_case = 0

    with torch.no_grad():

        for sentence, tags in prepared_data:

            tag_scores = model(sentence)
            selected_result = torch.max(tag_scores, dim=1)[1]
            diff = torch.abs(selected_result - tags)

            size = len(tags)
            correct_case += size
            total_case += size

            if (sum(diff) != 0):
                wrong_case = torch.sum((diff != 0).int())
                correct_case -= wrong_case

    return float(correct_case) / float(total_case)

prepared_test_data = [prepare_xy(test_sentence) for test_sentence in  test_data]
```


```python
import signal
import matplotlib.pyplot as plt
from IPython import display


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs, outputs = prepare_xy(training_data[0])
#     tag_scores = model(inputs)
#     print(tag_scores)


cnt = 0
loss_trace = []
accuracy_trace = []
smooth_loss = 0

total_epoch = 3
display_step = int(total_epoch * len(training_data) / 1000)
display_loss = 0.

prepared_training_data = [prepare_xy(tagged_sentence) for tagged_sentence in  training_data]


for epoch in range(total_epoch):
    try:
        for (sentence, tags) in prepared_training_data:

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # detaching it from its history on the last instance.
            model.hidden_LtoR = model.init_hidden()
            model.hidden_RtoL = model.init_hidden()

            # Step 3. Run our forward pass.
            tag_scores = model(sentence)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()

            display_loss = display_loss + float(loss)
            cnt += 1


            if (cnt % display_step == 0):

                ## display loss
                if (smooth_loss ==0):
                    smooth_loss = display_loss

                else:
                    smooth_loss = 0.99 * smooth_loss + 0.01 * display_loss


                loss_trace.append(smooth_loss)

                display.clear_output(wait=True)

                ## display accuracy 4 test data
                accuracy = get_accuracy(prepared_test_data)
                accuracy_trace.append(accuracy)

                plt.figure()
                plt.plot(accuracy_trace)
                plt.show()
                print('accuracy (test data):\t%f' % accuracy )

                plt.plot(loss_trace)
                plt.show()
                print('loss: %f' % (float)(display_loss/display_step))
                display_loss = 0            

    except KeyboardInterrupt:
        break
```


![Imgur](https://i.imgur.com/dSblHwM.png)


    accuracy (test data):	0.934136



![Imgur](https://i.imgur.com/332Od1I.png)


    loss: 0.135792


## 테스트 데이터로 실험하기


```python
prepared_test_data = [prepare_xy(test_sentence) for test_sentence in  test_data]
```


```python
def get_accuracy(prepared_data):
    correct_case = 0
    total_case = 0

    with torch.no_grad():

        for sentence, tags in prepared_data:

            tag_scores = model(sentence)
            selected_result = torch.max(tag_scores, dim=1)[1]
            diff = torch.abs(selected_result - tags)

            size = len(tags)
            correct_case += size
            total_case += size

            if (sum(diff) != 0):
                wrong_case = torch.sum((diff != 0).int())
                correct_case -= wrong_case

    return float(correct_case) / float(total_case)
```


```python
print('accuracy (training data):\t%f' %  get_accuracy(prepared_training_data))
print('accuracy (test data):\t%f' %  get_accuracy(prepared_test_data))
```

## 모델 저장하기


```python
torch.save(model.state_dict(), "bidirectional_learned.model")
```

## Word Embedding


```python
def get_embedding (word):
    return model.word_embeddings(word2id([word]))

king = get_embedding ('king')
queen = get_embedding ('queen')
man =  get_embedding ('man')
woman =  get_embedding ('woman')
```


```python
torch.dist(king-man+woman, queen)
```


```python
word_idx = list(range(len(unique_words)))
term = 'computer'
term_idx = word_to_idx[term]
word_idx.remove(term_idx)

long_tensor = torch.Tensor(word_idx).long()
term_idx = torch.Tensor([term_idx]).long()

embedded_words = model.word_embeddings(long_tensor)
embedded_term = model.word_embeddings(term_idx)
diffs = embedded_words - embedded_term
square_diffs = torch.mul(diffs, diffs)
square_sum = torch.sum(square_diffs, 1)
dist = torch.rsqrt(square_sum)

nearest_neighbor = (int)(torch.argmax(dist))

idx_to_word[nearest_neighbor]
#print(dist.max(dist))
```

## Reference

https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
