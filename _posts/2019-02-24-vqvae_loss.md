---
layout: post
title:  "VQ vae loss terms"
date:   2019-2-24
categories: [tensorflow]
mathjax: true
---


VQ-VAE의 로스는 3개의 term으로 이루어져 있다.


$$L = L_1 + L_2 + L_3$$


그리고 각각은 다음과 같이 정의된다.

$$L_1= \log{p(x | z_q(x))}\\
L_2 = \lVert \mathrm{sg}[z_{e}(x)] - e \lVert_{2}^{2}\\
L_3 = \beta \lVert z_{e}(x) - \mathrm{sg}[e] \lVert_{2}^{2}\\$$

L1은 디코더 아웃풋과의 비교를 통한 reconsturction loss이다.

Sonnet 의 VectorQuantize 클래스에서 계산하는 로스는 $L_2 + L_3$ 에 해당한다.

```python
vqloss = q_latent_loss + (self._commitment_cost * e_latent_loss) 
```
q_latent_loss = $L_2$  

논문에서 vq-objective로 정의하는 본 loss term 은 quantized embedding vector (e)가 encoder output (vq input)  과 가까워 지도록 하기 위함이다.

    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)

e_latent_loss  = $L_3$ (아직 commitment cost 상수는 안 곱한 것)

논문에서 commitment loss로 정의하는 본 loss term 은 embedding의 스케일을 regularize 하는 격이다 (뇌 해석). embedding을 배우는 것에 commit 하도록 한다. $L_3$ 는 commitment cost $\beta$ 와 e_latent_loss의 곱이다.

loss는 commitment cost 0.25 정도에서 robust (안정적이라는듯) 했음. 그러나 이는 $L_1$ 즉 reconstruction loss의 스케일에 따라 달라질 것이라고 함.



    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)


