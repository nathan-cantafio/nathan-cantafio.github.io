---
title: Pokémon Team Building as an Optimization Problem
date: 2025-12-20
description: A population-based optimization framework for Pokémon team building, using simulated battles to evaluate noisy black-box objectives over large combinatorial spaces.
---

### A Battle Agent Walks Into a Room

Here is something that actually happened during this project. I was watching a simulated Pokémon battle between two AI agents. One agent's Pokémon knew Dream Eater, a move that deals damage but only works if the target is asleep. The opposing Pokémon was wide awake. The agent used Dream Eater anyway. Then again. Then again. For twenty-four consecutive turns, it kept selecting Dream Eater against a fully conscious opponent, failing every single time, until the move ran completely out of uses. Only then did it switch to Psychic, which knocked out the opponent in two hits.

The opponent, for its part, had spent the entire battle using Agility (a move that raises your own speed) over and over again, to no strategic end.

I tell this story not to mock the agents, but because it perfectly captures what makes Pokémon team building such a strange optimization problem. You have a search space so large it defies comprehension. Your only way to evaluate a team is to simulate battles. And the battles themselves are controlled by agents that may spend two dozen turns doing something completely useless. What does it even mean to find a "good" team in this environment?

That question is what this project is really about.

### How I Got Here

I came across a YouTube video titled something like "Pokémon as a Machine Learning Problem" (it seems to have since been taken down). It argued that competitive Pokémon has two components: (1) team building and the (2) battles themselves. And that both are interesting machine learning problems.

The team building problem stuck with me. You have an astronomically large search space of possible teams. You want to find the best one. But "best" isn't well-defined independent of context: it depends on the opponents you'll face (which players call the metagame). And the most natural way to evaluate a team is to run it through as many battles as possible and compute a win-rate. But battles are stochastic (through critical hits, damage rolls, accuracy checks) so even the same team against the same opponent can go differently each time.

To recap the situation: team building is a *noisy black-box combinatorial optimization problem*. My immediate thought was Bayesian Optimization (BO), which is designed exactly for noisy black-box problems. But BO generally assumes a continuous search space, or at least one that can be reasonably embedded in $\mathbb{R}^d$. A Pokémon team is not continuous, and any sensible embedding would be very high dimensional. BO started to look less appealing.

I set that aside and turned to a more pressing question: how do you simulate Pokémon battles in the first place?

### Getting Battles Off the Ground

The standard approach in Python is `poke-env`, an interface that connects to a Pokémon Showdown server. The problem is that this requires all the overhead of a full Showdown server (including chat rooms, battle animations, player accounts) when really all I wanted was something that takes two teams, simulates a battle, and tells me who won.

So I tried writing my own battle engine. A few weeks in, I realized how many edge cases there were, and how hard it would be to verify that my engine was accurate to the original game. I went looking for existing engines and found one that was exactly what I wanted. Unfortunately it was written in Zig, and I could not get it to build.

I went back to `poke-env`. But I structured the project so that the battle engine is modular. That is, if I ever get another engine working, it should slot in without touching the optimization code.

### Formalizing the Problem

With battle simulation handled, I could set up the optimization problem properly.

To keep the search space tractable, I restricted the problem to **Generation 1**, where Pokémon have no held items, no abilities, no effort values, and no natures. Under these constraints, a team is fully specified by the choice of six species and four legal moves per species.

Even with this simplification, the search space is vast. Let me do the back-of-the-envelope calculation.

In Generation 1 there are 151 species. Choosing 6 without replacement, order not mattering, gives

$$\binom{151}{6} \approx 1.488 \times 10^9$$

possible species combinations. For each species, choosing 4 moves from an average learnset of around 30 gives

$$\binom{30}{4} = 27405$$

possible movesets. Putting it together:

$$\binom{151}{6} \cdot \binom{30}{4}^6 \approx 5.4 \times 10^{35}$$

distinct teams. To put that number in perspective: there are approximately $10^{19}$ grains of sand on earth. If you had one computer for each grain of sand evaluating one team per second since the earth formed 4.54 billion years ago, you would only just be finishing now.

The formal objective is the expected win-rate of a team $t$ against the metagame:

$$f(t) = \mathbb{E}[\text{win-rate} \mid t, \mathcal{M}]$$

where $\mathcal{M}$ is the distribution of opposing teams. In practice this is approximated by simulating a finite number of battles and computing the empirical win-rate.

This formulation has four properties that shape the choice of optimization method. The objective is a black box: there is no closed form, only simulation. It is stochastic: the same team can win or lose the same matchup on different runs. It is combinatorial: teams are discrete objects with constraints, so gradients are meaningless. And it is metagame-dependent: a team's strength is relative to its opponents, not absolute.

### Choosing an Optimization Strategy

Gradient-based methods are out immediately. Local search is suspect too: swapping a single Pokémon can cause large swings in performance because of matchup effects and team synergy, so the landscape is not smooth in any useful sense.

This points toward *population-based stochastic optimization*, which maintains a set of candidate solutions and iteratively improves them without making strong assumptions about the objective's structure. I structured the framework around two routines:

1. `evaluate_teams`: given a population of teams, assign each a score
2. `produce_next_generation`: given a scored population, generate the next one

This abstraction turned out to be the most useful design decision in the project. Once it was in place, swapping out optimization strategies became trivial. Random Search and Genetic Algorithms are just two different implementations of `produce_next_generation`.

For scoring, both methods use ELO ratings computed from battles within the population. ELO captures relative performance: beating a strong team earns more rating than beating a weak one, and losing to a strong team costs less than losing to a weak one. This lets the population build up a ranking without requiring a fixed set of opponents.

#### Random Search

In Random Search, `produce_next_generation` keeps the highest-rated teams from the current population and fills the rest with teams sampled uniformly at random from the legal search space. It is close to memoryless; the only continuity between generations is the small set of retained top performers.

#### Genetic Algorithms

In the Genetic Algorithm, `produce_next_generation` instead creates new teams by combining and modifying the top performers. Crossover takes two parent teams and produces a child that inherits some Pokémon from each. Mutation randomly replaces a Pokémon or alters one of its moves with some fixed probability. The idea is to bias exploration toward regions of the search space that have already shown promise, rather than sampling blindly.

#### Comparing the Two

Because ELO only measures relative performance within a population, ELO scores from different runs cannot be compared directly. A team that dominates a weak population will have high ELO, but might lose badly against a strong one.

This actually came up in the results in an interesting way. The best Random Search teams had higher ELO than the best Genetic Algorithm teams (but that turned out to be ELO inflation). Random Search generates a lot of very bad teams, which become "free ELO" for the mediocre teams that beat them. The population's floor was dragging the ceiling up artificially.

To get a fair comparison, I evaluated the final teams against a fixed gauntlet of 30 human-made OU teams and computed empirical win-rates. This is the actual metric.

### The Agent Problem

There is a subtlety I have been glossing over. A Pokémon battle is not determined by team composition alone. It also depends on the agents playing those teams. The objective function is really

$$f(t) = \mathbb{E}[\text{win-rate} \mid t, \pi, \mathcal{M}]$$

where $\pi$ is the battle agent policy. We are not finding "the best team" in some absolute sense. We are finding "the best team as played by a particular agent."

I used `poke-env`'s built-in `SimpleHeuristicPlayer`, which is a small step up from random play. This was a deliberate choice as  implementing a stronger agent would have been a significant additional project,  but it is a real limitation. A weak agent fails to exploit type matchups, switching, prediction, or long-term planning. The teams this project produces are not competitive in any human sense. They are teams that are robust under naive play.

The Dream Eater story at the top of this post is not an outlier. It is representative of how brittle the agent's decision-making can be. This matters because optimization aggressively exploits any bias in the evaluation pipeline. If the agent has blind spots, the optimizer will find teams that exploit those blind spots, not teams that would generalize to stronger play.

### Results

I ran both methods for 10 generations, (which is not very many; it would be preferable to run hundreds or thousands of generations). The ELO trajectories over time look like this:

<p align="center">
  <img src="/blog/assets/pokemon-team-opt/aggregate_performance.png" 
  alt="Aggregate ELO vs generation" style="max-width: 100%;">
</p>

And the win-rates against the gauntlet:

<p align="center">
  <img src="/blog/assets/pokemon-team-opt/aggregate_evaluation.png" 
  alt="Mean win rate vs gauntlet" style="max-width: 100%;">
</p>

Neither method does well in absolute terms after only 10 generations, but the Genetic Algorithm has a slight edge in win-rate. The ELO inflation effect is visible in the first plot: Random Search teams accumulate higher ELO despite performing worse on the gauntlet.

Below are GIFs showing how the best team (by ELO) evolved across generations for each method. I chose to show the highest-ELO team at each generation rather than the highest cumulative ELO, because ELO is a relative measure that changes meaning as the population changes. Tracking cumulative bests across shifting populations felt more misleading than informative.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="flex: 0 0 45%; text-align: center;">
    <img src="/blog/assets/pokemon-team-opt/team_evolution_EloGeneticAlgorithm.gif"
    alt="Team evolution (Genetic Algorithm)" style="max-width: 100%;">
  </div>
  <div style="flex: 0 0 45%; text-align: center;">
    <img src="/blog/assets/pokemon-team-opt/team_evolution_EloRandomSearch.gif"
    alt="Team evolution (Random Search)" style="max-width: 100%;">
  </div>
</div>

### What I Took Away

The win-rates are modest. But I think the more honest measure of the project is the framework.

Most of the work went into the evaluation pipeline () battle simulation, orchestration, bookkeeping) not into the optimization algorithms themselves. Once the `evaluate_teams` and `produce_next_generation` abstraction was clean, adding a new algorithm took almost no time. That kind of infrastructure is hard to build and easy to underestimate from the outside.

The bigger lesson is about what optimization actually does. It finds teams that score well under the specific evaluation setup you've built. If that setup has biases such as a weak agent or a particular metagame sample, then the optimizer will find those biases and exploit them. This is not a failure of the method. It is what optimization is supposed to do. The responsibility for making sure the evaluation reflects the thing you actually care about sits entirely with the designer.

### What Comes Next

The most obvious next step is a better battle agent. A stronger heuristic, or even a learned policy, would push the results toward something more strategically meaningful. An interesting extension would be co-evolution: jointly optimizing teams and agents, so that the agent's improving play creates pressure for the teams to improve too.

Beyond that, there are natural extensions on the algorithm side: more search strategies, direct win-rate evaluation against a fixed gauntlet rather than ELO, or a combinatorial BO approach. And while the code is built around Generation 1, extending it to later generations is at least conceivable, though it would require a significant refactor.

For now, I think of this project as a proof of concept. Pokémon team building can be framed cleanly as a noisy black-box combinatorial optimization problem, and even simple stochastic methods can make meaningful progress when the evaluation pipeline is carefully designed.

Thanks for reading!

### Code

The full code is on [GitHub](https://github.com/nathan-cantafio/pokemon).
