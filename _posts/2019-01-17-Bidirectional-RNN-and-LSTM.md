---
title: "Bidirectional RNN과 Bidirectional LSTM (이론편)"
layout: post
math: true
date: 2019-01-17
categories: NLP DeepLearning
permalink: nlp/deeplearning/Bidirectional-RNN-and-LSTM/
---

> 사이트 이전 후 링크가 박살이 나있는걸 2년만에 알았습니다. 송구합니다. 현재 수정 중이고, 이 페이지는 다 수정되었습니다. 죄송합니다.

> Google Analytics 설정도 잘못해놔서 아무도 제 블로그를 안보시는 줄알았습니다 ^^; 몰랐습니다. 죄송합니다 흑흑

이번 포스트에서는 Bidirectional Recurrent Neural Network (Bidirectional-RNN) 와 Bidirectional Long Short-Term Memory Network (Bidirectional LSTM)에 대해 알아보고 이를 PyTorch를 이용하여 직접 구현해본다.

## Preliminaries

- 본 포스트는 RNN 및 LSTM에 대한 사전지식이 필요함! 두 개념에 대해서는  지난 포스트에서 다룬 바 있음! (링크 참조)
    - [Vanilla RNN 실습]({{site.baseurl}}/nlp/deeplearning/Vanilla-RNN-%EC%8B%A4%EC%8A%B5/): Character-level language model with RNN
    - [Vanilla LSTM 실습]({{site.baseurl}}/nlp/deeplearning/Long-Short-Term-Memory-Network/): Character-level language model with LSTM

## 1. Motivation: 왜 Bidirectional인가?

[Word2Vec 논문 Review 포스트]({{site.baseurl}}/nlp/deeplearning/paperreview/Paper-Review-Distributed-Representations-of-Words-and-Phrases-and-their-Compositionality/#)에서 든 예문을 다시 살펴보자.

> 나는 ____를 뒤집어 쓰고 펑펑 울었다.

한국어를 자유자재로 사용하는 사람은 빈칸에 들어갈 말이 `이불`이라는 것을 쉽게 알 수 있다.

그런데 이 문장의 경우, 빈칸 유추 시 빈칸 앞보다는 빈칸 뒤에 나오는 단어들이 더 중요하다. '나는' 뒤에 나올 수 있는 단어는 수만개인 반면, '를 뒤집어 쓰고 펑펑 울었다' 앞에 나올 수 있는 단어는 흔치 않기때문이다. (조금 더 학술적으로 말하면, `이불`이 아닌 어떠한 단어 `w`에 대한 확률 `P(w|를 뒤집어 쓰고 펑펑 울었다)`는 `P(이불|를 뒤집어 쓰고 펑펑 울었다)`보다 매우 작다.)

이 예제에서 볼 수 있듯, 텍스트 데이터는 정방향(시점을 기준으로 과거에서 미래 방향) 추론 못지 않게 역방향(시점을 기준으로 미래에서 과거 방향) 추론도 유의미한 결과를 낼 수 있다. 하지만 일반적인 RNN 구조는 오로지 정방향으로 데이터를 처리한다.

![](https://i.imgur.com/7Eyz4WY.png)

위 그림에서 볼 수 있듯, RNN은 과거 (t = 0, 1, ..., (i-1) ) 시퀀스를 읽어 현재 (t=i) 데이터 에 대한 레이블을 예측하는 것은 가능한 구조이다. 그러나 역방향의 연결은 존재하지 않기 때문에 현시점보다 미래 시점인 데이터는 추론 시 활용할 수 없다.

이러한 단점을 해소하고자 나온 개념이 연결이 양방향으로 존재하는 RNN인 [Bidirectional RNN [1]](https://ieeexplore.ieee.org/abstract/document/650093/)이다.



## 2. Bidirectional RNN 의 구조

![Imgur](https://i.imgur.com/XGpeu82.png)





Bidirectional RNN은 위 그림처럼 생겼다. 보다시피 Hidden unit에는 두 종류 (녹색, 주황색)가 있다. 녹색 뉴런들은 정방향 (왼쪽에서 오른쪽으로, 또는 과거에서 미래 시점으로)으로 연결되어있으며, 주황색 뉴런들은 역방향(오른쪽에서 왼쪽으로, 또는 미래 시점에서 과거시점으로)으로 연결되어 있다. 이 둘을 concatinate하면 완전한 i번째 hiddden layer 값인 $$hs[i]$$가 완성된다.

Training의 경우도 RNN의 Backpropagation과 크게 다르지 않다. Loss가 전달되는 path가 무엇인지 유의하며 식을 유도하면 쉽게 알 수 있다.

$$\overleftarrow{W_{xh}}$$의 update rule만 유도해보자.

![Imgur](https://i.imgur.com/Ekp90q0.png)




## 3. Bidirectional LSTM Network로 개념 확장하기

Bidirectional RNN이 이해되었다면 Bidirectional LSTM Network은 간단하다. Bidirectional RNN은 방향이 서로 다른 두 RNN이 만들어낸 i번째 hidden neuron들을 concatinate하여 i번째 입력에 대한 히든 레이어로 삼았다. Bidirectional LSTM Network은 여기에서 RNN대신 LSTM Network를 사용한다는 것만 다르다. 구조는 아래 그림과 같다.

![Imgur](https://i.imgur.com/fLc4u4w.png)

## 4. Bidirectional LSTM Network으로 Part-of-Speech Tagging 해보기

[링크]({{site.baseurl}}/nlp/deeplearning/Pos-Tagging-with-Bidirectional-LSTM/)로 이동!

## Reference

[1] Schuster, Mike, and Kuldip K. Paliwal. "Bidirectional recurrent neural networks." IEEE Transactions on Signal Processing 45.11 (1997): 2673-2681.
