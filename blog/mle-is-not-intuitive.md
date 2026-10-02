---
title: MLE is not intuitive
date: 2025-12-13
description: The justification for using MLE estimates seems fairly intuitive, but this post makes an argument for why it isn't.
---

Maximum likelihood estimation (MLE) feels natural. It isn't, and the reason it works is not the reason it feels natural.

You have independent data $x_1, \dots, x_n$ from a family of distributions with density $f(x|\theta)$ and unknown $\theta$. The **likelihood** is

$$L_n(\theta) = \prod_{i=1}^n f(x_i|\theta),$$

and the MLE is $\hat\theta = \arg\max_{\theta \in \Theta} L_n(\theta)$. The story is: pick the parameters under which the data were most likely.

That story quietly treats likelihood as probability. For continuous models it isn't. It is a product of density values, and densities can be arbitrarily large. A normal density at its mean blows up as the variance shrinks. So why doesn't MLE chase degenerate parameters to drive the likelihood to infinity? Usually it doesn't. The question is why.

### The real justification: regularity conditions

Under "regularity conditions," the MLE converges in probability to the true $\theta^\*$ as $n \to \infty$. The typical conditions:

1. **Identifiability:** different parameters give different distributions.
2. **A well-separated maximum:** parameters far from $\theta^\*$ have meaningfully smaller expected log-likelihood.
3. **A uniform law of large numbers** for the log-likelihood: random fluctuations don't create spurious peaks.

With these, and assuming the model contains the truth, the MLE is consistent. With more assumptions it is asymptotically normal, which gives the usual confidence intervals and tests.

Courses usually present this backwards: motivate MLE with "most likely data," then add the regularity conditions as a technical footnote. But "most likely data" is not a justification for continuous models, since density values are not probabilities. The asymptotic theory is where the justification lives. The intuition is a mnemonic for what MLE *does*. It is not a reason to believe MLE *works*.

### When the conditions fail: Gaussian mixtures

Consider a two-component mixture,

$$p(x)=\pi_1 N(x|\mu_1,\sigma_1^2)+\pi_2 N(x|\mu_2,\sigma_2^2),$$

with $\pi_1 + \pi_2 = 1$ and both weights strictly between $0$ and $1$. Set $\mu_2 = x_n$. That data point's second-component term is

$$\pi_2 N(x_n|x_n,\sigma_2^2)=\frac{\pi_2}{\sqrt{2\pi\sigma_2^2}} \to \infty \quad\text{as } \sigma_2\to 0,$$

while the first component fits the remaining points. The likelihood is unbounded, so the maximum doesn't exist [1]. This violates the well-separated maximum condition.

A single Gaussian doesn't have this problem. Set the mean to $x_n$ and shrink the variance, and the other points' contributions go to $0$ fast enough to drag the whole likelihood to $0$. In the mixture, the other component absorbs those points, and the degenerate component is unchecked.

Practitioners know this. Standard mixture-fitting code adds a small constant (like `1e-6`) to the covariance diagonal. That replaces the ill-posed problem with a nearby well-posed one. The regularity conditions fail for the original problem and are *enforced* by the modification.

### A finite-sample view

Asymptotics is the real justification, but there is also a finite-sample question: for fixed $n$, how likely is a sample that makes the likelihood huge? Let $X_1, \dots, X_n \sim P_{\theta^\*}$ and define

$$A_N = \left\\{\sup_{\theta \in\Theta} L_n(\theta) > N \right\\}.$$

For a well-behaved model we want $\mathbb{P}(A_N) \to 0$ as $N \to \infty$. For the Gaussian mixture, $\mathbb{P}(A_N) = 1$ for every $N$.

**Bounded densities.** If $f(x|\theta) < M$ for all $x, \theta$, then $L_n \le M^n$ and $A_N$ is empty for $N \ge M^n$. Gaussians are not bounded, but restricting $\sigma^2 > \varepsilon^2$ bounds the density by $1/\sqrt{2\pi\varepsilon^2}$. This is what the `1e-6` does.

**Exponential.** Let $X_i \sim {\rm Exp}(\lambda^\*)$, so $f(x|\lambda)=\lambda e^{-\lambda x}$. With $S = \sum_i X_i$, the MLE is $\hat\lambda = n/S$ and

$$\sup_\lambda L_n(\lambda) = \left(\frac{n}{eS}\right)^n.$$

This exceeds $N$ exactly when $S < n/(e N^{1/n})$. Since $S \sim \mathrm{Gamma}(n, \lambda^\*)$, we have $\mathbb{P}(S < s) \le (\lambda^\* s)^n / n!$, so

$$\mathbb{P}(A_N) \le \frac{1}{N}\cdot\frac{(\lambda^\* n/e)^n}{n!}.$$

**Gaussian.** Let $X_i \sim \mathcal{N}(\mu^\*, {\sigma^\*}^2)$ with $n \ge 2$. Evaluating at the MLE gives

$$\sup L_n = (2\pi e\,\hat\sigma^2)^{-n/2},$$

so $A_N$ requires $\hat\sigma^2 < (2\pi e)^{-1} N^{-2/n}$. Since $\hat\sigma^2 \ge R^2/(2n)$, where $R$ is the sample range, this forces $R < \sqrt{n/(\pi e)}\, N^{-1/n}$. Conditioning on $X_1$, each other point must land within $R$ of it, which has probability at most $2R/(\sqrt{2\pi}\sigma^\*)$. Hence

$$\mathbb{P}(A_N) \le \left(\frac{c_n N^{-1/n}}{\sigma^\*}\right)^{n-1}, \qquad c_n = \frac{1}{\pi}\sqrt{\frac{2n}{e}},$$

which goes to $0$ as $N \to \infty$.

In both cases, huge likelihoods need the sample to be squeezed into a tiny region, and the true distribution gives that region vanishing probability. In the mixture, nothing needs to be squeezed, so the likelihood is unbounded for every sample.

### Final remarks

These examples suggest $\mathbb{P}(A_N) \to 0$ whenever the classical regularity conditions hold. I suspect this is true but haven't tried to prove it.

The point stands regardless. MLE is principled: consistency under regularity conditions is a real result. But "choose the parameters that make the data most likely" and the asymptotic theory are different justifications, and statistics education often presents the first as if it were the second. The regularity conditions are usually a footnote. They are the headline.

Thanks for reading!

### References

[1] Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.
