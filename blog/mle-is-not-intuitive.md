---
title: MLE is not intuitive
date: 2025-12-13
description: The justification for using MLE estimates seems fairly intuitive, but this post makes an argument for why it isn't.
---

When I was first introduced to maximum likelihood estimation (MLE), it felt like a very natural and intuitive idea.

Here is the setup. You have independent data, say $n$ observed values $x_1, \dots, x_n$. You believe these came from some family of distributions, parameterized by some unknown $\theta$. For example, you might believe your data came from a normal distribution, and $\theta = (\mu, \sigma^2)$ are the unknown mean and variance. The goal is to estimate $\theta$ from the data.

MLE says: choose the $\hat\theta$ that makes the observed data most "likely." More precisely, if the distribution has density $f(x|\theta)$, define the **likelihood function**

$$L_n(\theta) = \prod_{i=1}^n f(x_i|\theta),$$

which is just the joint density of all observations, treated as a function of $\theta$. The MLE is then

$$\hat\theta = \arg\max_{\theta \in \Theta} L_n(\theta).$$

If you quietly identify "likelihood" with "probability," this feels completely reasonable: choose the parameters under which the data were most likely to occur.

The problem is that, for continuous distributions, likelihood is not a probability at all. It is a product of density values evaluated at the observed data points. And unlike probabilities, densities can be made arbitrarily large. For example, for a normal distribution, shrinking the variance concentrates more and more mass near the mean, causing the density at that point to blow up.

This raises an immediate question: if densities can be made arbitrarily large, why doesn't MLE collapse to degenerate parameter values (like a variance going to zero) in order to drive the likelihood to infinity? In practice this rarely seems to happen. MLE usually produces sensible estimates, even for continuous models.

So what is really going on?

### "Regularity Conditions" — The Real Justification

The principled justification for MLE comes from consistency theory. Under a set of "regularity conditions," one can prove that as the sample size $n \to \infty$, the MLE converges in probability to the true parameter $\theta^\*$. Typical conditions include:

(1) **Identifiability:** different parameter values correspond to genuinely different distributions;

(2) **A well-separated maximum:** parameters far from the true value attain meaningfully smaller likelihood in expectation;

(3) **A uniform law of large numbers for the log-likelihood:** random fluctuations in data do not create spurious likelihood peaks in the long run.

Under these conditions (and assuming the true data-generating process belongs to the model) the MLE is consistent. With additional assumptions, one also obtains asymptotic normality, which is the basis for standard confidence intervals and hypothesis tests built on top of MLE.

I want to be clear: this is a genuinely good justification. Consistency is not a trivial property. It tells you that MLE is not just a heuristic and that there is a precise sense in which maximizing the likelihood is the right thing to do.

But here is the subtle point I want to make. These conditions are often presented in courses as a kind of afterthought, a box to check after MLE has already been motivated on intuitive grounds. The typical pedagogical arc goes: "MLE is natural because you're choosing the parameters that make the data most likely. Oh, and here are some technical conditions under which it's consistent." The intuition is treated as the main course and the regularity conditions as a perfunctory afterthought.

I would argue this has it backwards. The intuitive story: "pick the parameters that make the data most likely" is not really a justification at all for continuous models, because likelihood is not probability. A density value is not a probability. You can make densities arbitrarily large. The intuitive framing only works if you already have some reason to believe that maximizing this particular quantity leads somewhere sensible, and that reason is precisely the asymptotic theory.

The regularity conditions are the meat. The intuition is a useful mnemonic, but it is not where the justification lives.

### When the Regularity Conditions Fail: Gaussian Mixture Models

If the regularity conditions are the real justification for MLE, it's worth asking what actually happens when they fail. A canonical example is the Gaussian mixture model; and it's instructive precisely because it arises in a model that statisticians use every day.

Consider a mixture of two normal distributions with unknown means and variances: $$p(x)=\pi_1 N(x|\mu_1,\sigma_1^2)+\pi_2 N(x|\mu_2,\sigma_2^2),$$

where $\pi_1+\pi_2=1$, both weights are strictly between $0$ and $1$, and $N(x|\mu,\sigma^2)$ is the Gaussian density with mean $\mu$ and variance $\sigma^2$.

Now suppose we set the mean of the second component exactly equal to one of the data points $x_n$. That data point contributes to the likelihood as:

$$N(x_n|x_n,\sigma_2^2)=\frac{1}{\sqrt{2\pi\sigma_2^2}},$$

which tends to infinity as $\sigma_2\to 0$. Meanwhile, the first component can be used to fit the remaining data. So the likelihood is unbounded. Maximizing it is not a well-posed problem. [1]

This is a concrete failure of the regularity conditions, specifically the well-separated maximum: there is no maximum at all.

Contrast this with a *single* Gaussian. If you try the same trick of setting the mean equal to one data point and shrinking the variance, then the contribution from that point does blow up, but the contributions from all *other* data points all tend to $0$ fast enough that the joint likelihood goes to $0$ overall. The pathology is tamed by the rest of the data.

In the mixture case, this self-correction fails because the other component can absorb the remaining observations, leaving the degenerate component free to blow up unchecked.

This failure is well-known in practice. Standard algorithms for fitting Gaussian mixture models therefore impose explicit or implicit regularization (for instance, preventing covariance matrices from becoming singular by adding a small constant (like `1e-6`) to the diagonal). In effect, practitioners quietly replace the ill-posed maximum likelihood problem with a nearby, well-behaved one. The regularity conditions don't hold for the original problem; they are *enforced* by the modification.

### A Finite-Sample Perspective: Why the Conditions Hold When They Do

The asymptotic theory is the right justification for MLE, but it leaves open a more tactile question: for a fixed, finite sample, what is actually preventing the likelihood from blowing up? This doesn't replace the asymptotic argument. Rather, it helps explain why the regularity conditions are reasonable assumptions in the first place, by showing that the pathological samples they rule out are intrinsically unlikely under the true distribution.

More precisely, I want to ask:

> **For a "nice" data model, what is the probability that a sample of fixed size $n$ leads to a degenerate likelihood:  one that can be made arbitrarily large?**

More precisely, fix $n \in \mathbb{N}$, and let $X_1,\dots,X_n \sim P_{\theta^\*}$ come from a parameterized family. Let $L_n(\theta)$ be the joint likelihood. Define

$$A = \left\\{(X_1,\dots, X_n) : \sup_{\theta \in\Theta} L_n(\theta) = \infty \right\\},$$

the set of samples for which the likelihood can be made arbitrarily large. What is $\mathbb{P}(A)$?

#### A Simple Sufficient Condition

If the density $f(x|\theta)$ is uniformly bounded, that is, there exists a constant $M$ (not depending on $x$ or $\theta$) such that $f(x|\theta) < M$ for all $x$ and $\theta$. Then for any finite sample and any $\theta$,

$$L_n(\theta) = \prod_{i=1}^n f(x_i|\theta) \leq M^n < \infty.$$

So $\mathbb{P}(A) = 0$ trivially.

This is a strong assumption. Gaussians, for instance, do not satisfy it: the density at the mean grows without bound as the variance shrinks. However, we can restore it by simply restricting the parameter space. If we enforce $\sigma^2 > \varepsilon^2$ for some $\varepsilon > 0$, then the density is bounded above by $\frac{1}{\sqrt{2\pi\varepsilon^2}}$. This is essentially what practitioners do when they add `1e-6` to covariance matrices.

#### Two Examples in the Unbounded Case

What about models where the density is unbounded, and we don't restrict the parameter space? Here the story is more subtle, but still reassuring.

**1. Exponential distribution.** Let $X_1,\dots, X_n\sim{\rm Exp}(\lambda^\*)$ with density $f(x|\lambda)=\lambda e^{-\lambda x}$ for $x\geq 0$, $\lambda>0$. The joint likelihood is

$$L_n(\lambda)=\lambda^n\exp\left(-\lambda\sum_{i=1}^n X_i\right),$$

and the MLE is $\hat\lambda = n / \sum_{i=1}^n X_i$. If all observations fall very close to $0$, say $X_i < \varepsilon$ for all $i$, then $\sum X_i < n\varepsilon$ and $\hat\lambda > 1/\varepsilon$, so

$$\sup_{\lambda > 0} L_n(\lambda) > \left(\frac{1}{\varepsilon}\right)^n e^{-n}.$$

Taking $\varepsilon = 1/N$ shows the likelihood can exceed $N^n e^{-n}$, which is enormous for large $N$. So degenerate samples do exist.

But how likely are they? A standard calculation gives

$$\mathbb{P}\left(\max_{1\leq i\leq n} X_i < \varepsilon\right) = \left(1 - e^{-\lambda^\* \varepsilon}\right)^n \approx (\lambda^\* \varepsilon)^n$$

for small $\varepsilon$, via a Taylor expansion. This is vanishingly small. Degenerate samples exist, but they are extremely unlikely under the true distribution.

**2. Gaussian distribution.** Let $X_1, \dots, X_n \sim \mathcal{N}(\mu^\*, {\sigma^\*}^2)$ with density

$$f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).$$

The MLEs are the sample mean $\hat{\mu} = \bar X$ and sample variance $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$. If all observations are concentrated in an interval of length $\varepsilon$ — i.e., $\max_{i,j}|X_i - X_j| < \varepsilon$, then $\hat\sigma^2 < \varepsilon^2$, and evaluating the likelihood at the MLE gives

$$L_n(\hat\mu,\hat\sigma^2) > \left(2\pi\varepsilon^2\right)^{-n/2}\exp\left(-\frac{n}{2}\right),$$

which tends to infinity as $\varepsilon\to 0$.

Again, though, the probability of this event shrinks extremely rapidly. The event $B_\varepsilon = \{\max_{i,j} |X_i - X_j| < \varepsilon\}$ satisfies

$$\mathbb{P}(B_\varepsilon) \leq \left(\Phi\left(\frac{\varepsilon}{2\sigma^\*}\right) - \Phi\left(-\frac{\varepsilon}{2\sigma^\*}\right)\right)^n \approx \left(\frac{\varepsilon}{\sqrt{2\pi}\,\sigma^\*}\right)^n,$$

where $\Phi$ is the standard normal CDF. This goes to zero extremely quickly as $\varepsilon \to 0$.

The pattern is the same in both examples: samples that make the likelihood blow up are concentrated in an extremely small region of the sample space, and they are assigned vanishingly small probability under the true distribution.

### Some Final Remarks

The examples above suggest a more general result might be provable: that $\mathbb{P}(A) = 0$ whenever the classical regularity conditions hold. I suspect this is true but haven't tried to prove it carefully.

More broadly, I think the point I've been circling deserves to be stated plainly. MLE is a principled method: consistency under regularity conditions is a real and non-trivial result. But the intuitive justification ("choose the parameters that make the data most likely") and the principled justification (asymptotic theory) are not the same thing, and statistics education often presents the former as though it were the latter. The intuition is a helpful way to remember what MLE *does*; it is not a reason to believe MLE *works*.

The regularity conditions are usually introduced as a footnote. I'd argue they are the headline.

Thanks for reading!

### References

[1] Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.
