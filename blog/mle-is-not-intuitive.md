---
title: MLE is not intuitive
date: 2025-12-13
description: The justification for using MLE estimates seems fairly intuitive, but this post makes an argument for why it isn't.
---

When I was first introduced to maximum likelihood estimation (MLE), it felt like a very natural and intuitive idea. The first distributions I learned about were discrete, with probability mass functions that really did assign probabilities to events. Then I learned about continuous distributions, which had densities. Of course, it was emphasized that you cannot evaluate the density at a point to obtain the probability of realizing that point. But my rough intuition was to think of probability densities as a kind of proxy for probabilities. That intuition mostly worked well enough.

Then came MLE. Given a parameterized family of distributions, we estimate the parameters by choosing the ones that maximize the likelihood of the observed data. If you quietly identify “likelihood” with “probability,” this feels completely reasonable: choose the parameters under which the data are most likely to occur. If anything my introduction to MLE re-enforced my rough inuition about densities.

The problem is that, for continuous distributions, likelihood is not a probability at all. It is a product of density values evaluated at the observed data points. And unlike probabilities, densities can be made arbitrarily large. For example, in a normal model, shrinking the variance concentrates more and more mass near the mean, causing the density at that point to blow up.

This raises a question. If densities can be made arbitrarily large, why doesn’t maximum likelihood estimation collapse to degenerate parameter values. Such as a variance going to zero in order to drive the likelihood to infinity? In practice, this rarely seems to happen. MLE usually produces sensible estimates, even in continuous models.

So what is really going on? Why does maximizing the likelihood work so well, and under what conditions does it fail?


### "Regularity Conditions"

In my experience, many statistics courses mention some often under-explained “regularity conditions” under which the MLE has nice properties. For example, one typically assumes

(1) identifiability: different parameter values correspond to genuinely different distributions in the model;

(2) a well-separated maximum: even parameter values that are very close to the MLE will attain smaller likelihood

(3) a uniform law of large numbers for the log-likelihood: roughly speaking, this ensures that random fluctuations in the data do not create spurious likelihood peaks in the long run.

Under these assumptions, one can guarantee that the MLE converges in probability to the true parameter value. Of course, it is also generally assumed that the true data-generating process actually belongs to the model class. With some additional conditions, one can further obtain asymptotic normality of the estimator.

At first glance, these regularity conditions seem to save us. Though they do not prevent individual densities from becoming arbitrarily large, they do ensure that, with enough data, doing so comes at a significant cost to the joint likelihood of all observations. 

Still, I find this explanation unsatisfying. These results are only asymptotic: they tell us that with enough data, degenerate likelihood behaviour won't be an issue, but they say very little about how much data is “enough” in practice. Moreover, there are common examples in which these regularity conditions fail outright, and yet maximum likelihood estimation still gets used (albeit with some tweaks) and works well!

### When the Regularity Conditions Fail: Gaussian Mixture Models

A canonical example where these regularity conditions fail is the Gaussian mixture model. Consider a mixture of two normal distributions with unknown means and variances. Namely $$p(x)=\pi_1N(x|\mu_1,\sigma_1^2)+\pi_2N(x|\mu_2,\sigma_2^2),$$ where $\pi_1+\pi_2=1$ and are both less than 1, and $N(x|\mu,\sigma^2)$ is the density of Gaussian distribution with mean $\mu$ and variance $\sigma^2$. If the second component has its mean exactly equal to one of the data point $x_n$, this data point will then contribute to the likelihood as a term like: $$N(x_n|x_n,\sigma_2^2)=\frac{1}{\sqrt{2\pi\sigma_2^2}}.$$ Which tends to infinity as $\sigma_2\to 0$. The first mixture component can fit the remaining data. Thus in this case the maximization of the likelihood is not well-posed. [1]

The behaviour reflects a failure of the regularity conditions. In particular, the likelihood does not have a well-separated maximum; in fact it has no maximum at all.

In contrast this problem does not occur with a single Gaussian. If a single Gaussian sets its mean exactly equal to one  data point, then even though that data point's contribution to the likelihood does tend to infinity as $\sigma\to0$, the remaining data points contribution all tend to $0$ fast enough so that the joint likelihood tends to $0$ overall.

This example illustrates that it's possible for the joint density itself to become arbitrarily large. In practice, this problem is well-known. Thus standard methods for performing MLE on Gaussian mixture models impose (either implicitly or explicitly) regualarization, such as ensuring covariance matrices are invertible  by adding a small amount like `1e-6` to the diagonal, or other modifications. These modifications restore the boundedness of the likelihood ensuring the regularity conditions are met. In effect, practitioners replace the ill-posed maximum likelihood problem with a similar well-behaved one. (For a Gaussian mixture models there is another issue of identifiability that I don't want to get into).

### A Finite Sample Probabilistic Approach

But in order to satisfy my doubts about MLE, I want to try and give a satisfactory answer to the admittedly vague question: 

> **For a 'nice' data model (one that satisfies certain regularity conditions), what is the probability that a sample of fixed sized $n$ leads to a 'degenerate' likelihood (a joint likelihood which can be made arbitrarily large for certain choices of parameters).**

Formulating this a little bit more precisely. Fix $n\in\mathbb{N}$, and let $X_1,\dots,X_n\sim P_{\theta^*}$ come from a family of distributions parameterized by $\theta\in\Theta$. Let $L_n(\theta)$ be the joint likelihood of the $n$ observations. Let $$A_m = \left\\{(X_1,\dots, X_n):\sup_{\theta \in\Theta} L_n(\theta)>m\right\\}$$ and let $$A=\bigcap_{m\in \mathbb{N}} A_m=\left\\{(X_1,\dots, X_n:\sup_{\theta\in\Theta}L_n(\theta)=\infty)\right\\},$$ so that $A$ is the set of samples for which the likelihood can be made arbitrarily large. What is $\mathbb{P}(A)$?

#### A restricted setting

If we assume that the distribution $P_\theta$ has a density $f(x|\theta)$ which is continuous and uniformly bounded in the sense that there is a constant $M$ (not depending on $x$ or $\theta$) such that $f(x|\theta)<M$ for all $x,\theta$, then for any finite independent sample of realizations $x_1,\dots, x_n$ and $\theta\in \Theta$ we have  $$L_n(\theta)=\prod_{i=1}^n f(x_i|\theta)\leq\prod_{i=1}^n M=M^n<\infty.$$ So $\mathbb{P}(A)=0$.

This is quite a strong assumption. For example we already saw that a Gaussians do not satisfy this property. If $n=1$ (or if $n>1$ and all observations are the same though this has probability $0$) then setting the mean to $x_1$ and letting the variance tend to $0$ allows the likelihood to be arbitrarily large. However we can fix the situation by enforcing $\sigma^2>\varepsilon^2$ for some $\varepsilon>0$. In this case we can take $M = \frac{1}{\sqrt{2\pi \varepsilon^2}}$.

#### Examples in a more general setting

We often  seem to have issues when all of our observations are too close to each other. Let's look at two simple examples:

1. Let $X_1,\dots, X_n\sim{\rm Exp}(\lambda^*)$ be independent with density given by $$f(x|\lambda)=\lambda e^{-\lambda x},\qquad x\geq 0,\lambda>0.$$ The joint likelihood is $$L_n(\lambda)=\lambda^n\exp\left(-\lambda\sum_{i=1}^n X_i\right).$$ The MLE is $$\hat\lambda = \frac{n}{\sum_{i=1}^n X_i}.$$ Now let $\varepsilon>0$. If all observations are concentrated near $0$, that is $X_i<\varepsilon$,  then $\sum X_i < n\varepsilon$ and so $\hat\lambda > 1/\varepsilon$. Hence $$\sup_{\lambda >0} L_n(\lambda)>\left(\frac{1}{\varepsilon}\right)^n e^{-n}.$$ If we take $\varepsilon=1/N$ for some $N\in\mathbb{N}$ we see that $\sup_{\lambda >0}L_n(\lambda)>N^n e^{-n}$ which I think makes the fact the likelihood can be made very large more salient. 

    However, it is a standard calculation that 

    $$\mathbb{P}(X_1<\varepsilon,\cdots, X_n<\varepsilon)=\mathbb{P}\left(\max_{1\leq i\leq n}X_i<\varepsilon\right)=\left(1-e^{-\lambda^*\varepsilon}\right)^n.$$

    Which via a Taylor expansion of $e^t$ for small $\varepsilon$ this is approximately $(\lambda^*\varepsilon)^n$ which is vanishingly small. 
    
    So while there are samples that lead to "degenerate likelihoods" they are very unlikely samples indeed.

2. Let $X_1, \dots, X_n \sim \mathcal{N}(\mu^*, {\sigma^*}^{2})$ be independent with density given by  
   $$f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad \sigma^2 > 0.$$  
   The joint likelihood is  
   $$L_n(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(X_i - \mu)^2}{2\sigma^2}\right) = (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2 \right).$$  
   The MLEs are  
   $$\hat{\mu} = \frac{1}{n} \sum_{i=1}^n X_i=\bar X, \quad \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (X_i - \hat{\mu})^2.$$

   Now let $\varepsilon > 0$. If all observations are concentrated within a small interval of length $\varepsilon$, that is  
   $$\max_{1 \leq i,j \leq n} |X_i - X_j| < \varepsilon,$$  
   then the sample variance is bounded by $$\hat\sigma^2 = \frac1n\sum_{i=1}^n(X_i-\bar X)^2<\varepsilon^2.$$ Thus, $$\sum_{i=1}^n (X_i-\bar X)^2<n\varepsilon^2.$$ The likelihood at $\mu=\hat\mu$ and $\sigma^2=\hat\sigma^2$ is $$L_n(\hat\mu,\hat\sigma^2)=\left(2\pi\hat\sigma^2\right)^{-n/2}\exp\left(-\frac{1}{2\hat\sigma^2}\sum_{i=1}^n (X_i-\bar X)^2\right)>\left(2\pi\varepsilon^2\right)^{-n/2}\exp\left(-\frac{n}{2}\right).$$ Which tends to infinity as $\varepsilon\to 0$.

   However, the probability of all observations falling within such a small interval shrinks rapidly as $\varepsilon$ gets small. More precisely, the event  
   $$B_\varepsilon = \left\\{(X_1, \dots, X_n) : \max_{i,j} |X_i - X_j| < \varepsilon \right\\}$$  
   can be bounded by

   $$\mathbb{P}(B_\varepsilon) \leq \mathbb{P}\left(\max_{1\leq i\leq n} |X_i - \mu^\star| < \frac{\varepsilon}{2}\right) = \left\(\Phi\left(\frac{\varepsilon}{2\sigma^*}\right) - \Phi\left(-\frac{\varepsilon}{2\sigma^\star}\right)\right)^n$$
     
   where $\Phi$ is the standard normal CDF. For small $\varepsilon$, this probability is approximately  
   $$\left(\frac{\varepsilon}{\sqrt{2\pi} \sigma^*}\right)^n,$$  
   which tends to zero very quickly as $\varepsilon \to 0.$

   Thus, while samples that enable the likelihood to blow up exist, they become increasingly unlikely for finite $n$ as the threshold $\varepsilon$ shrinks.


### Some Final Remarks

I wonder if a more general result can be proven, for example that $\mathbb{P}(A)=0$ so long as the classic regularity conditions are satisfied.

This point about densities not being probabilities and so why should maximizing them be a good idea is a subtle one. And I feel that it gets  glossed over in statistics education. Maybe that's because in the end everything works out nicely so it doesn't really matter. But I do feel this was worthwhile thinking about.

Thanks for reading!

### References

[1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
