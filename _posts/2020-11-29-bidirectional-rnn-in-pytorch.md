---
title: "PyTorch에서의 Bidirectional RNN에 대한 정확한 이해"
layout: post
math: true
date: 2020-11-28
permalink: bidirectional-rnn-in-pytorch/
---

> 본 포스트는 [Understanding Bidirectional RNN in PyTorch- Ceshine Lee](https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66)를 한국어로 번역한 자료입니다.

> This post is a Korean translation version of the post: [Understanding Bidirectional RNN in PyTorch](https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66) - by Ceshine Lee.

![](https://miro.medium.com/max/1313/1*6QnPUSv_t9BY9Fv8_aLb-Q.png)

Fig 1. Bidirectional Recurrent Neural Networks의 일반적인 구조. ([출처:colah’s blog](http://colah.github.io/posts/2015-09-NN-Types-FP/))


Bidirectional recurrent neural networks(RNN)은 두 개의 독립적인 RNN을 서로 합친 모델이다.
입력 시퀀스는 한 RNN에 대해 정방향으로 입력되고, 다른 RNN에 대해 역방향으로 입력된다.
두 RNN의 출력은 각 time step 기준으로 concat되어 최종 출력이 결정된다. (또는 concat 대신 더하는 등의 variation이 있다.)
이러한 구조는 네트워크로 하여금 각 time step별로 정방향과 역방향 정보를 모두 갖게된다.
이러한 개념 자체는 그다지 어려운 것이 아니지만. 이걸 실제로 이용하려고 하다보면 약간 헷갈리는 부분이 생긴다...


## 헷갈리는 부분

첫 번째로 헷갈리는 부분은 bidirectional RNN의 출력을 어떻게 Dense Layer (Fully-connected Layer, Linear Layer)로 전달하냐는 것이다.
일반 RNN의 경우 마지막 time step의 출력 값을 활용하면 된다. bidirectional RNN에서의 비슷한 기법을 구글링을 통해 알아보자.


![](https://miro.medium.com/max/741/1*GRQ91HNASB7MAJPTTlVvfw.jpeg)

Fig 2. 혼란스러스운 구조 ([출처](http://doc.paddlepaddle.org/develop/doc/_images/bi_lstm.jpg))

뭐..뭐지? 위에 그림처럼 마지막 time stamp의 아웃풋만 이용하게 된다면 역방향 RNN은 마지막 인풋 (역방향  RNN입장에서는 첫번째 인풋인 $x_3$)만 보게된다.
이러한 경우 역방향 RNN은 예측적인 기능을 전혀 제공하지 못한다.

> 역자 주: 직관적으로 생각해보아도, 첫 번째 인풋만 처리한 결과는 역방향 시퀀스를 대표하는 값이 될 수 없지 않은가?

두 번째로 헷갈리는 부분은 hidden state에 대한 것이다.

> 역자 주: python ```output, hidden = rnn(input)``` 임을 기억해보자.  ```output```과는 다르게, ```hidden``` 의 경우 각 layer 별로 결과가 출력되므로 ```output```과는 다른 관점에서 혼란스러운 점이 발생하여 분리하여 기술한 것으로 생각됨.  

seq2seq 모델에서 디코더의 hidden state 을 초기화하기 위해서는 인코더의 hidden state 반환 결과를 알아야한다.
직관적으로 생각해보면, PyTorch에서와 같이 특정 time step에서 hidden state를 고른다면,
RNN이 가장 마지막 시퀀스 인풋을 처리한 상태의 hidden state이 필요할 것이다.
그러나 아까 그림에서 보았다시피, 마지막 단계의 hidden state을 사용하게 되면, 역방향 RNN입장에서는 하나의 입력만 보게되는 문제가 발생한다.


## Keras의 구현을 확인해보자.

Keras에서는 bidirectional RNN에 대한 wrapper class (API를 제공하는 껍데기)가 구현되어있다.

```wrappers.py```의 [line 292](https://github.com/keras-team/keras/blob/4edd0379e14c7b502b3c81c95c7319b5df2af65c/keras/layers/wrappers.py#L292) 을 살펴보면

역방향 RNN의 출력이 default로 역방향 정렬되는 것을 확인할 수 있다.

```python
if self.return_sequences:
            y_rev = K.reverse(y_rev, 1)
```

Keras는 ```return_sequences```이 ```true```일 때 (default로는 false)만 역방향으로 정렬해준다.
- 따라서 특정 time step ```n```의 값을 취한다면 Keras는
	- 정방향 RNN에 대해서는 ```n```번째 인풋
	- 역방향 RNN에 대해서는 ```1```번째 인풋
을 취하게 된다.

이렇게 보자면 figure 2가 좀 결함있는 구조를 가지고 있음을 알 수 있다.

## PyTorch에서는 어떨까?

첫 번째 헷갈리는 부분은 다소 해결된 것 같다.
이제 PyTorch에서 bidirectional RNN을 제대로 사용하는 방법에 대해 알아보자.

<script src="https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1.js"></script>

위 notebook 코드는 두 가지 헷갈리는 부분에 대한 답을 가지고 있다. (단, ```batch_first```는 false라고 가정)

1. (output에 관하여) 우리는 정방향 RNN에 대해 ```output[-1, :, :hidden_size]```, 역방향 RNN에 대해 ```output[0, :, hidden_size:]```를 취해야하고, 이를 concat한 후, 그 결과를 dense neural network에 전달해야한다.
2. (hidden state에 관하여) 아웃풋되는 hidden state는 전체 시퀀스를 모두 처리한 결과이다. 이들은 디코더에 안전하게 전달되어도 된다.

---

### Side note

Pytorch GRU의 [ouput 형태](http://pytorch.org/docs/master/nn.html#torch.nn.GRU)는 ```batch_first```가 flase일때

- output (seq_len, batch, hidden_size * num_directions)
- h_n (num_layers * num_directions, batch, hidden_size)

---

LSTM의 경우도 비슷하나, 추가적으로 h_n과 같이 shaped된 Cell state 도 반환된다는 점이 다르다.
