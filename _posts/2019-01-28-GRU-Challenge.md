---
title: "Hackathon: Character-level language model with GRU"
layout: post
math: true
date: 2019-01-28
categories: NLP DeepLearning
permalink: nlp/deeplearning/GRU-Challenge
---

이번 포스트에서는 GRU(Gated Recurrent Unit)에 대해 알아본다. GRU란 [1]에서 소개된 개념으로 Long Short-Term Memory Cell과 같이 장기적 기억이 가능하면서도 계산량은 절감시킨 RNN cell이다. RNN과 LSTM에 대한 내용은 이전 포스트 ([RNN](https://ws-choi.github.io/nlp/deeplearning/Vanilla-RNN-%EC%8B%A4%EC%8A%B5/), [LSTM](https://ws-choi.github.io/nlp/deeplearning/Long-Short-Term-Memory-Network/))를 참조할 것!

[실습 코드 Link! (연구실에서만 이동 가능)](http://163.152.111.187:8001/tree/Hackathon%20-%20GRU%20Challenge)
r
## Challenge

```
1. 더 작은 Loss 값을 만들 것
2. 더 적은 Iteration을 사용할 것
3. 더 적은 Parameter를 사용할 것
4. 굳이 GRU나 LSTM을 사용하지 않아도 되며 다계층 RNN (LSTM, GRU 포함) 활용도 가능
```

## Preliminaries: RNN

![](https://i.imgur.com/7Eyz4WY.png)

RNN 구조를 다시 한번 복습해보자!


## 1. GRU의 구조

t번째 hidden unit인 h_{t}를 산출하기 위한 GRU의 수식은 다음과 같다.

![Imgur](https://i.imgur.com/88iUZf5.png)

$$
\begin{aligned}
z_{t}         &= \sigma ( W_{z}x_{t} + U_{z}h_{t-1} + b_{z} ) \\
r_{t}         &= \sigma ( W_{r}x_{t} + U_{r}h_{t-1} + b_{r} ) \\
\tilde{h_{t}} &= \tanh (W_{h}x_{t} + U_{h}(r_{t} \odot h_{t-1}) + b_{h}) \\
h_{t} &= ( \begin{bmatrix} 1 \\ 1\\ ... \\ 1 \end{bmatrix} -z_{t})  \odot h_{t-1} + z_{t} \odot \tilde{h_{t}}
\end{aligned}
$$



## 2. GRU를 이용한 Character-level Language Model 만들기

본 실습의 목적은 GRU를 이용하여 주어진 문장(sentence)을 Character 단위로 모델링하는 Language Model을 만드는 것이다. 이를 더 Formal하게 표현해보자.

훈련하고자하는 모델의 파라미터 $$ \theta = [ W_{z}, U_{z}, b_{z}, W_{r}, U_{r}, b_{r}, W_{h}, U_{h}, b_{h}   ] $$ 와
주어진 문장 데이터 집합 $$S = \{ s_{1}, s_{2}, s_{3}, ... , s_{n}\}$$에 대해

$$\prod_{s \in S} P(s | \theta) $$

를 최대화시키는 $$ \theta $$ 를 찾는 것이 목표이다.

### 2-1. Loss Function 도출하기

위에서 설정한 목적을 토대로 Loss Function을 설계해보자.

$$
\begin{aligned}
        \mathcal{L} &= - \frac{1}{|S|}  \sum_{s \in S} log P(w | \theta) \\
        &=   - \frac{1}{|S|} \sum_{s=c_{1}c_{2}...c_{m} \in S}  log P( c_{1} | \theta) \times P( c_{2} | c_{1}, \theta) \times ... \times  P( c_{m} | c_{1}c_{2}...c_{m-1}, \theta) \\
        &=   - \frac{1}{|S|} \sum_{s=c_{1}c_{2}...c_{m} \in S}   (\prod_{c_{i} \in s} log P( c_{i} | c_{1}, c_{2},..., c_{i-1}, \theta) \\
        &=   - \frac{1}{|S|} \sum_{s=c_{1}c_{2}...c_{m} \in S} ( \sum_{c_{i} \in s} log (P( c_{i} | c_{1}, c_{2},..., c_{i-1}, \theta) \\
    \end{aligned} %]]>
$$




## 3. Training Data - Anser Data 만들기


```python
import numpy as np

data = open('input.txt', 'r', encoding='UTF-8').read()

unique_chars = list(set(data))
unique_chars.append('SOS')
unique_chars.append('EOS')

vocab_size = len(unique_chars)
print('num of chars: %d' % vocab_size)

sentences = data.split('\n')

char_to_ix = { ch:i for i,ch in enumerate(unique_chars) }
ix_to_char = { i:ch for i,ch in enumerate(unique_chars) }

def encoding (text):
    encoded_text = [char_to_ix[ch] for ch in text]
    return encoded_text

def decoding (text):
    decoded_text = [ix_to_char[ix] for ix in text]
    return decoded_text

training_set = [ encoding(sent) for sent in sentences]
for t_data in training_set:
    t_data.insert(0, char_to_ix['SOS'])

answer_set    = [ encoding(sent) for sent in sentences]
for a_data in answer_set:
    a_data.append(char_to_ix['EOS'])

print('------------------\nTraining Data - Anser Data Example: ')
print('\tTraining Data\t:', ''.join(decoding(training_set[0])))
print('\tAnswer Data\t:', ''.join(decoding(answer_set[0])))
```

    num of chars: 52
    ------------------
    Training Data - Anser Data Example:
    	Training Data	: SOSA statistical language model is a probability distribution over sequences of words.
    	Answer Data	: A statistical language model is a probability distribution over sequences of words.EOS


## 4. 모델 정의


```python
def init_parameters (D, H, var=0.01):
    raise NotImplementedError
```


```python
def sigmoid (X):
    return 1./(1. + np.exp(-X))

def tanh (X):
    return np.tanh(X)

def forward (X, Htm1, _params):

    Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wy, by = _params
    raise NotImplementedError    

    return ZtTilde, RtTilde, Zt, Rt, ZtBar, HtrawTilde, HtTilde, H_t, Y, P
```


```python
def get_derivative (_params, inputs, targets, hprev):

    Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wy, by = _params
    Xs, ZTeldes, RTeldes, Zs, Rs, ZBars, HrawTildes, HTildes, Hs, Ys, Ps = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    Hs[-1] = np.copy(hprev)
    loss = 0

    # forward
    for t in range(len(inputs)):
        Xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        Xs[t][inputs[t]] = 1        
        ZTeldes[t], RTeldes[t], Zs[t], Rs[t],  ZBars[t], HrawTildes[t], HTildes[t], Hs[t], Ys[t], Ps[t] = forward(Xs[t], Hs[t-1], _params)

        loss += -np.log(Ps[t][targets[t], 0])

    dWz, dWr, dWh = np.zeros((H, D)), np.zeros((H, D)), np.zeros((H, D))
    dUz, dUr, dUh = np.zeros((H, H)), np.zeros((H, H)), np.zeros((H, H))
    dbr, dbz, dbh = np.zeros((H, 1)), np.zeros((H, 1)), np.zeros((H, 1))
    dWy, dby      = np.zeros((D, H)), np.zeros((D, 1))

    dHnext = np.zeros_like(hprev)

    # backward
    for t in reversed(range(len(inputs))):

        # Phase 5
        raise NotImplementedError

        # Phase 1
        raise NotImplementedError

        # Phase 2
        raise NotImplementedError

        # Phase 3
        raise NotImplementedError

        # Phase 4
        raise NotImplementedError

        # Phase 5
        raise NotImplementedError

        np.clip(dHnext, -5, 5, out=dHnext)

    return  loss, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWy, dby

```

## 5. 학습시켜보기


```python
def guess_sentence_GRU (_params, max_seq = 150):

    initial_char = char_to_ix['SOS']
    x = np.zeros((D, 1))
    x[initial_char] = 1

    h = np.zeros((H, 1))

    ixes = []

    n=0
    while True:

        output = forward (x, h, _params) # ZtTilde, RtTilde, Zt, Rt, ZtBar, HtrawTilde, HtTilde, H_t, Y, P
        p = output[-1] # probabilities for next chars
        ix = np.random.choice(range(D), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        h = output[-3]

        n+=1
        if ( n > max_seq):
            break
        elif (ix == char_to_ix['EOS']):
            break

        ixes.append(ix)

    return ''.join(decoding(ixes))
```


```python
import signal
import matplotlib.pyplot as plt
from IPython import display

learning_rate = 0.1

loss_trace = []


def optimize(iteration, D, H) :

    m = len(training_set)

    _params = init_parameters(D, H)
    mems = []
    for param in _params:
        mems.append(np.zeros_like(param))

    for n in range(iteration):
        loss_sum = 0

        for inputs, targets in zip(training_set, answer_set):

            hprev = np.zeros((H,1))

            loss, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWy, dby = get_derivative (_params, inputs, targets, hprev)
            loss_sum += loss

            for param, dparam, mem in zip(_params,
                                          [dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWy, dby],
                                         mems):
                dparam = np.clip(dparam, -5, 5)
                mem += dparam * dparam
                param += -1./m * learning_rate * dparam/np.sqrt(mem + 1e-8) # adagrad update


        if((n+1) % 10 == 0):
            loss_trace.append(loss_sum/m)
            display.clear_output()
            plt.figure()
            plt.plot(loss_trace)
            plt.show()
            print(loss_sum/m)

            for i in range(5):
                print(guess_sentence_GRU(_params))

    return _params


iteration = 1000
D = vocab_size
H = 100
Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, Wy, by = optimize(iteration, D, H)
```


![Imgur](https://i.imgur.com/ireCN3p.png)


    301.3838469688061
    md1oci strr.ngyur
    F'afmsehnygifeaasnuiaooresceao e mnre iaan e wioirdlnih  ncgiorhadolo p ghmhmor tal ioono n thunetnr uaptrvedn befnbe Aap ppeamo  bmo eai son wdd  l i
    u—agteecgimrvpranhd iilmawld,ie lmeehrsaj"mnluoucoahioeee ao  rnuse  
    vLw r ree dmsoytsdca dmhsttdqlgaf
    eae.dt  lwrs c "h c svopite1 sleeciot  hu sn redm-ysseo r nlsqegcs aes cis eem.slnae i  ithnsntx nbs.uaeeorti scaooihionhaemsisr rlvius gw tiejn re-el


## Reference.

[1] Cho, Kyunghyun, et al. "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation." Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2014.
