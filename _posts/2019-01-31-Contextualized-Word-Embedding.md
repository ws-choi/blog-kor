---
title: "Contextualized Word Embedding (개념편)"
layout: post
math: true
date: 2019-01-31
categories: NLP DeepLearning PaperReview
permalink: nlp/deeplearning/paperreview/Contextualized-Word-Embedding/
comments: true

---

본 포스트에서는 Contextual Word Embedding 개념에 대해 설명한다. ELMo[1], Bert[2]등의 Contextual Word Embedding 기술들의 디테일한 비교는 다음 포스트에서 다룰 예정이며, 본 포스트에서는 전통적인 Word Embedding 기법인 Word2Vec과의 비교를 통해 Contextualized Word Embedding의 개념에 대해서만 설명한다.

## Contextualized Word Embedding?

Contextualized Word Embedding은 단어를 저차원 (일반적으로 100~500 차원) 공간에서 표현하는 기법이다. 단, 기존의 전통적 Word Embedding과는 달리, 같은 단어더라도 문맥에 따라 그 표현방법이 바뀔 수 있는 개념의 Embedding이다. ELMo[1], Bert[2]등의 Contextual Word Embedding 기법은 최근 각종 NLP task의 벤치마크 기록을 갱신하고 있다. 일례로, 최근 한국어를 대상으로하는 QA Dataset인 [KorQuAD(The Korean Question Answering Dataset)](https://korquad.github.io/)의 순위권 기법은 거의 모두Bert를 기반으로 하고 있다. 이와 관련하여, [카카오 자연어 처리팀 박상길님의 포스트](http://docs.likejazz.com/bert/)에는 Bert에 대한 분석과 PyTorch를 이용한 Bert 기반 서비스(KorQuAD 포함) 구현에 대한 내용이 아주 상세하게 잘 설명되어 있다.

![Imgur](https://i.imgur.com/0veMIjE.png)

그렇다면, 이제부터 본격적으로 Contextualized Word Embedding에 대해 알아보자!


## 1. Introduction: 빈칸 추론 문제

[Word2Vec 포스트]({{site.baseurl}}/nlp/deeplearning/paperreview/Paper-Review-Distributed-Representations-of-Words-and-Phrases-and-their-Compositionality/)에서 다루었듯, 언어에 대한 전반적인 이해를 평가하는 데에는 빈칸 추론 문제만한 것이 없다. 이전 포스트의 예제를 다시 살펴보자. 한국어에 대한 전반적인 이해가 높은 사람은 다음과 같은 빈칸 추론 문제를 쉽게 풀 수 있을 것이다.

> 나는 ___을 뒤집어 쓰고 펑펑 울었다

한국말을 잘하는 사람들은 빈칸에 들어갈 말로 ‘이불’이 적합하다는 것을 어렵지 않게 유추할 수 있다. 매우 익숙한 단어의 시퀀스이기 때문이다. 그렇다면 만약, 아래 그림처럼 빈칸이 뚫린 문맥을 입력으로 받고 그 빈칸에 들어갈만한 단어를 맞추는 (즉, 예측하는) 모델이 있다고 가정해보자.

![]({{site.baseurl}}/assets/postimages/Untitled-5da34d59-8bcc-4404-93d3-fc93cb2d4e16.png)

만약 엄청난 양의 데이터를 학습한 위 빈칸 추론 모델이 사람에 버금가는 수준으로 빈칸 추론 문제를 잘 푼다고 가정해보자. 그렇다면 적어도 이 빈칸 추론 모델은 단어 사용 패턴에 있어서는 사람만큼의 지식을 갖추고있다고 볼 수 있지 않을까?

## 2. Pre-trained Word Representation의 시작: NNLM과 Word2Vec

지난 포스트에서 다룬 [Neural Probability Lanuage Model]({{site.baseurl}}/nlp/deeplearning/paperreview/Paper-Review-A-Neural-Probabilistic-Language-Model/) (또는 Neural Network Language Model)에서도 이러한 빈칸 추론 문제를 이용하여 Word에 대한 Distributed Representation을 만들었다. 그리고 이 과정을 단순화시키되, 엄청나게 많은 데이터를 학습하여 Word Embedding이라는 혁신을 불러온 논문이 [Word2Vec]({{site.baseurl}}/nlp/deeplearning/paperreview/Paper-Review-Distributed-Representations-of-Words-and-Phrases-and-their-Compositionality/#)이다. 두 연구의 공통점은 다음과 같다.

- 빈칸 추론 문제에 기반한 Loss function을 만들고, 이를 최소화하는 pre-training 과정을 통해 Word에 대한 Distributed Representation을 얻는다.
- 그렇게 얻은 Word Embedding은 기존 NLP Task (Machine Translation, Named Entity Recognition 등) 문제를 해결한다.

그렇다면 이번에는 두 기법의 차이점을 정리해보자.

![Imgur](https://i.imgur.com/DcYOcTx.png)

디테일한 관점에서 다양한 차이점이 있지만, 이번 포스트에서 주목하는 것은 pre-training 시 어떠한 Loss Function을 기반으로 했는가이다. NNML의 경우 전통적인 통계적 언어 모델 (Statistical Language Model)을 계승했다고 볼 수 있다. 즉, NNLM의 목표는 다음 likelihood를 최대화 시키는 Parameter set을 찾는 것이다. (특히 그 중에서도 Word Embedding Matrix를 찾는 것이 핵심)

$$ f(x_t,...,w_{t-n+1}) = \hat{P}(w_{t} | x_{t-1},...,w_{t-n+1})$$

주어진 데이터들에 대한 위 likelihood 값이 높아지는 방향으로 NNLM이 학습되게 된다.

반면 Word2Vec의 경우, 주변 단어들로 가운데 단어를 예측(CBoW)하거나, 가운데 단어로 주변 단어들을 예측하는 방법을 택한다. 여담으로, [3]논문에서도 나와있듯, Word2Vec는 전통적인 통계적 언어 모델의 계승을 포기한다.

> While NCE can be shown to approximately maximize the log probability of the softmax, the Skipgram model is only concerned with learning high-quality vector representations, so we are free to simplify NCE as long as the vector representations retain their quality

즉, Skip-gram의 경우 확률론적인 해석이 불가능할지라도 낮은 계산량으로 양질의 Word Representation만 얻을 수 있으면 장땡이라는 것이다.

어쨋든, 이렇게 해서 만들어진 Pre-trained Word Representation 기법들은 단어가 가지는 문법적, 의미적 정보를 100~500 차원의 실수공간에서 잘 잡아냈다. 이를 단적으로 보여주는 증거가 [word vector간의 벡터 산술[3, 5]]({{site.baseurl}}/nlp/deeplearning/paperreview/Glove/)이다.

> Word2Vec(‘‘king")−Word2Vec(‘‘man")+Word2Vec(‘‘woman")∼Word2Vec(‘‘queen")
![](https://adriancolyer.files.wordpress.com/2016/04/Word2Vec-plural-relation.png?w=600)
Image Sorce: https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/

또한 실제 NLP에 적용하였을 때 기존의 기법 (One-Hot-Encodig 등) 보다 월등히 좋은 성능을 보여주었기에 지난 몇년동안 NLP의 새로운 표준으로 떠올랐다.

## 3. 기존 Pre-trained Word Representation 한계

단어를 더 이상 고차원 공간의 sparse한 점으로 표현하는 것에서 벗어나 저차원의 dense vector로 표현했으며, 그렇게 표현된 점이 단어의 문법적, 의미적 정보까지 잡아내는 데에 성공했다는 점에서 Word2Vec은 기념비적인 논문이다. 그러나 이러한 기법도 개선할 여지가 있다며 나온 개념이 있다. 바로 Contextualized Word Embedding 개념이다. 그렇다면 Contextualized Word Embedding를 개발한 연구자들은 전통적인 Word Embedding의 어떤 점이 마음에 안들었을까? 다음 예제를 보자.


> (1) 나는 ```이불```을 뒤집어 쓰고 펑펑 울었다.

> (2) ```이불```(泥佛)은 흙으로 빚어 만든 불상을 뜻합니다


```이불```은 동음이의어이기 때문에 (1)과 (2)에서의 의미가 다르다. 그러나 Word2Vec 모형에서는 (1)에서의 이불이나 (2)에서의 이불이나 같은 점으로 Embedding될 것이다. 이것 뿐만이 아니다.

> (3) 올해는 ```이불``` 하나 장만해야죠...

> (4) 새로산 ```이불```이 너무 따듯하다.

> (5) ```이불``` 밖은 위험해

> (6) 집가서 ```이불```킥할듯


(3) ~ (6)에서는 ```이불```이 (1)과 같은 사전적 의미의 ```이불```이다. 그런데 이 경우에도 문맥에 따라 ```이불```이라는 단어가 가지는 뉘앙스가 미묘하게 다르다. 만약 (4)를 들은 사람이 '극세사 이불인가요?'라고 질문한다면 매우 정상적인 대화겠지만 (1)을 듣고 '극세사 이불인가요?'라고 답하면 사회성없다고 욕먹을 수 있다.

> **사족**: 세미나에서 이 예제를 이용하여 발표하였더니, 재미있게도 세대별로 (3)~(6)의 이불의 뜻에 대한 이해도가 달랐다. 각 예제에서의 ```이불```의 뜻을 보충 설명하자면, (3)의 경우는 실제 이불 장만을 뜻할 수도 있지만, ```결혼```을 암시하는 경우가 종종 있다. 이불로 예단을 하는 경우가 많기에 경험이 없는 어린 사람일수록 잘 모르는 용례. 반대로 (5)~(6)은 한때 많이 사용되던 유행어인데, 각각 ```집에서 꼼짝도 안하고자하는 의지```과 ```창피함```을 뜻하는 용례이다. (5)의 경우 철지난 유행어인 ```방콕```의 뜻에 가깝다.


여기서 알 수 있는 사실은 같은 단어라도 문맥에 따라 쓰임새가 다를 수 있다는 것이다. 역으로 이러한 미묘한 뉘앙스를 구분해낼 줄 안다면 해당 언어에 대한 이해도가 높다고 평가할 수 있다. 그런 면에서 전통적인 Pre-trained Word Representation은 언어에 대한 이해도가 다소 떨어진다고 할 수 있다. (1) ~ (6)까지의 이불의 쓰임새는 모두 다름에도 불구하고, 이들을 모두 같은 vector로 표현하니 말이다. 물론 word embedding 뒤에 나오는 모델이 그 뉘앙스 차이를 간파할 수 있을지도 모른다. 그러나 만약 Word Embedding 레벨에서 자체적으로 단어에 대한 높은 이해도를 재현해낸다면, NLP의 성능을 더 높일 수 있지 않을까?

## 4. Contextualized Word Representation

Contextualized Word Embedding은 같은 단어라도 문맥에 따라 다른 vector를 만들어낸다. 대표적으로 ELMo[1], Bert[2], OpenAI GPT 등의 기법이 있다. 이들의 특징은 같은 단어라도 문맥에 따라 다른 방식으로 표현(representation)한다는 것이다. 또한 이러한 작업을 위해 매우 Deep 한 신경망을 사용한다.

그렇다면 Contextualized Word Embedding이 어떻게 문맥에 따라 달라지는 단어의 의미를 잡아내는지 아주 개념적으로만 살펴보자.

![Imgur](https://i.imgur.com/eRIYjpZ.png)

위의 그림은 Word2Vec의 Embedding 과정이며, 아래의 그림은 Contextualized Word Representation의 동작 과정이다. 단, 위 아래 그림 모두 pre-training을 마치고 실제로 단어를 embedding하는 과정에 대한 그림이지, 훈련 과정을 묘사하고 있는 그림이 아니다.

Word2Vec의 Embedding은 단어 단위로 이루어진다. 즉, 각각의 단어에 대한 one-hot-encoding vector가 $$W^T$$와 곱해져서 Word Vector를 얻게된다. (물론 효율문제로 인해 실제로는 one-hot-encoding vector를 곱하는 방식을 사용하지는 않음)

반면 Contextualized Word Representation의 경우 문장을 받아 각 단어에 대한 representation을 산출해준다. 가장 밑단에서는 각 단어에 대해 가장 기초적인 형태의 embedding ($$e_{t}^{[1]}$$)을 산출한다. 이 과정은 Word2Vec과 거의 유사하다. 그러나 그 이후 여러 레이어를 거쳐 문장을 구성하는 서로 다른 단어 사이의 의미교환이 이루어진다. 이 과정을 통해 '문맥'에 의존적인 '단어의 의미'를 잡아내는 feature가 산출된다.

두 기법의 차이를 표로 정리해보자.

| 관점 | Word2Vec | Contextualized Word Representation|
| --- | --- | ---|
| Input | 단어 단위 | 문장 단위 (단어의 시퀀스)|
| Layer | (일반적으로) 단층 | (일반적으로) 다계층|
| Output | 해당 단어에 대한 Embedding | 문장을 구성하는 각 단어에 대한 Embedding들|



이번 포스트에서는  Word2Vec과의 비교를 통해  Contextualized Word Representation의 연구동기 및 강점을 개념적으로 살펴보았다. 다음 포스트에서는 Contextualized Word Representation을 위한 다양한 기법들을 살펴보고 각 기법을 비교해볼 예정이다.


> **사족:** 엄밀히 말하면, ElMo를 제외한 나머지 기법, 특히 Bert가 Contextualized Word Representation을 위한 기법인가 아닌가에 대해서는 이견이 있을 수 있다. [6]과 같은 Survey Paper에서는 Bert를 Contextualized Word Embedding으로 분류한다. 반면 Bert[2]의 저자진은 Bert의 Contextualized Word Representation의 기능보다는 Bert의 Trensfer Learning적 기능을 더 강조하는 듯하다. 즉, Feature Extraction의 기능보다는 supervised data로 fine-tuning하기 전에 많은 양의 un-labeled 데이터를 pre-training하여, 언어에 대한 전반적 이해도가 매우 뛰어난 모델을 만들 수 있는 역할 그 자체로 Bert를 바라보는 듯 하다. 그래서인지 제목에서도 Word Representation이라고 표현 대신 Language Understanding이라는 표현을 사용하고있다. 본 포스트에서는 Bert 또한 내부적으로 문맥에 따라 달라지는 단어 (또는 token)의 embedding을 추출하는 과정을 포함한다는 관점에서 Bert 등의 기법을 Contextualized Word Embedding로 분류한다.


## Reference

[1] Peters, Matthew, et al. "Deep Contextualized Word Representations." Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers). Vol. 1. 2018.

[2] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

[3] Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.

[4] Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of machine learning research 3.Feb (2003): 1137-1155.

[5] Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.

[6] Young, Tom, et al. "Recent trends in deep learning based natural language processing." ieee Computational intelligenCe magazine 13.3 (2018): 55-75.
