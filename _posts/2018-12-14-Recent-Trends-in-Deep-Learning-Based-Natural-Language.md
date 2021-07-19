---
title: "Recent Trends in Deep Learning Based Natural Language Processing"
layout: post
math: true
date: 2018-12-18
categories: NLP DeepLearning PaperReview
permalink: nlp/deeplearning/paperreview/Recent-Trends-in-Deep-Learning-Based-Natural-Language
---

최근 딥러닝을 기반으로 하는 NLP(Natural Language Processing: 자연어처리)나 Text Mining 기법들이 Top Conference에 많이 투고되고 있다. 현재 우리 연구실에서도 이러한 트렌드를 쫓아가려고 최신 연구동향 Survey paper

> Young, Tom, et al. "Recent trends in deep learning based natural language processing." ieee Computational intelligenCe magazine 13.3 (2018): 55-75.

를 분석하고 있다. 운 좋게도 이기창 선배님 블로그 [ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/)에 해당 논문이 한글로 잘 번역되어 있어서 페이퍼 읽는 시간을 줄일 수 있었다.

본 블로그는 이 페이퍼와 [ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/)를 읽으며 정리하면서 쓰는 노트이다.

주의: 본 블로그에는 서베이 페이퍼 및 rasgo 블로그 이외에도 많은 개인적 추측과 사족을 달고 있습니다. 또한 아직 작성 중인 블로그입니다.

---

## Abstract

다 읽고 나서 쓰겠습니다. TBC

## Introduction

자연어처리(NLP:Natural Language Processing, 이후 NLP)와 Deep Learning 등에 대한 연구 배경이 나온다. Introduction의 본론만 간추려서 말하자면

> 최근 Dense vector representation 기반의 신경망 기법들은 다양한 NLP task에서 두각을 드러내고 있다. 이러한 트랜드는 word embedding [2,3] 과 deep learning [4] 기법의 성공 덕에 촉발되었다. 그러니 NLP에 사용되는 다양한 딥러닝 모델 및 기법에 대해 살펴보겠다.

로 요약할 수 있다.

## Distributed Representation

Distributed Representation은 왜 필요하며, 어떤 기법들이 있는가가 이 챕터의 핵심이다.

- Distributed Representation은 왜 필요한가?
이 논문에서는 딥러닝 이전의 NLP 기법을 통계적 NLP (Statistical NLP)로 통칭하나보다. 여튼 이 전통적인 NLP 기법은 [차원의 저주 (curse of dimensionality)](https://en.wikipedia.org/wiki/Curse_of_dimensionality) 문제가 발생한다고 한다. [7]번 논문에서 설명되어있다는데 내용이 아주 길어서 안읽었다. 일단 받아드리고 나중에 읽어야지.
- Distributed Representation에는 어떤 기법이 있는가?
    - Word Embeddings
    - Word2Vec (Word Embedding의 일종)
    - Character Embeddings

## Distributed Representation - Word Embeddings

Word Embedding 기법은 '단어'를 '*d-*차원의 vector'로 embedding (mapping과 비슷하게 해석하면 될 듯함) 시키는 기법을 총칭하는 듯하다. 그러나 단어를 단순하게 vector 형식으로 표현하는 기법을 거창하게 distributed representation 이라고 부르는 것 같진 않다. [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model)도 단어를 벡터로 보낸다는 점에서 보면 Word Embedding이지 않은가?  서베이 논문에서는, Distributional vector 또는 word embedding은 다음 핵심 가정 ([Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis))을 따른다고 한다.

- [**Distributional Hypothesis**](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis): 같은 문맥에서 사용되거나 나타나는 (occur) 단어들은 유사한 의미를 가지는 경향이 있다.

> Words that are used and occur in the same contexts tend to purport similar meanings (Wikipedia)

(사족: 사실 이 정의는 서베이 논문에서 나온 표현은 아니고, 위키피디아 표현이다. 이 표현이 본 게시글의 논리 전개에 더 적합한 듯하여 위키피디아 표현을 빌렸다.)

그렇다면 이 가정을 도대체 어디다가 쓰는가? 서베이 논문에서는 입문자가 이해하기에는 다소 불친절하하게 설명되어 있다. 문제의 문장을 살펴보자.

> Thus, these vectors try to capture the characteristics of the neighbors of a word.

여기서 읽다가 좀 막혔다. Word embedding 기법이 Distributional Hypothesis를 기반으로 하는 것과, Word embedding 기법이 추구하고자 하는 바 (직역: 벡터들은 주변 단어의 특성을 담아내야한다.) 가 무슨 상관이란 말인가? 입문자인 내게는 다소 비약처럼 느껴졌다. 개인적인 추측을 더해 서베이 논문에서의 매우 압축된 논리 전개에 살을 붙여 진술해보겠다.

1. **기법에서 사용하는 핵심가정**
    - 같은 문맥에서 사용되는 단어들은 유사한 의미를 가지는 경향이 있다.
2. **은근슬쩍 끼워놓은 가정**
    - 문맥을 관찰 가능한 텍스트, 즉 단어의 sequence라고 정의하자.
3. **은근슬쩍 입맛대로 바꾼 핵심가정**
    - 서로 다른 두 단어 `w1`와 `w2`가 사용되는 텍스트들을 조사했을 때 주변에서 나타나는 단어들의 패턴이 비슷하다면, 두 단어의 뜻도 유사할 것이다.
4. **암묵적으로 추구해야 하는 Embedding의 성질**
    - 서로 다른 두 단어 `w1`와 `w2`가 비슷한 의미로 사용된다면, 이 둘을 embedding시켰을 때의 벡터값이 유사해야한다.
5. **결론**
    - 4를 만족시키려면 비슷한 의미를 가진 서로 다른 두 용어를 embedding 시켰을 때의 벡터값도 유사하도록 유도해야한다.
    - 이를 위해서는 타겟 단어의 주변에서 나타나는 다른 단어들의 패턴을 조사해야한다.
    - 즉, 타겟단어의 주변 단어들의 특징을 조사해야해야 암묵적으로 추구해야하는 embedding의 성질을 만족하며 Distributional Hypothesis를 따르는 embedding이 가능하다.
