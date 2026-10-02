---
title: Pokémon Team Building
date: 2025-12-20
description: A population-based optimization framework for Pokémon team building, using simulated battles to evaluate noisy black-box objectives over large combinatorial spaces.
---

### A Battle Agent Walks Into a Room

During this project I watched a simulated battle between two AI agents. One agent's Pokémon knew Dream Eater, a move that only works if the target is asleep. The opponent was wide awake. The agent used Dream Eater anyway, for twenty-four consecutive turns, until the move ran out of uses. Only then did it switch to Psychic, which knocked out the opponent in two hits. The opponent, meanwhile, spent the whole battle using Agility, a move that raises its own speed, to no end.

This captures what makes Pokémon team building a strange optimization problem. The search space is incomprehensibly large. The only way to evaluate a team is to simulate battles. And the battles are controlled by agents that may waste two dozen turns. What does a "good" team even mean here?

### How I Got Here

I came across a YouTube video, "Pokémon as a Machine Learning Problem" (since taken down), arguing that competitive Pokémon has two components: team building and the battles themselves, and that both are interesting ML problems.

Team building stuck with me. The search space is astronomically large. "Best" isn't defined independent of the opponents you'll face (the metagame). The natural evaluation is win-rate over many battles, but battles are stochastic (critical hits, damage rolls, accuracy), so the same matchup can go differently each time.

So team building is a *noisy black-box combinatorial optimization problem*. My first thought was Bayesian Optimization (BO), which is built for noisy black-box problems. But BO assumes a continuous space, or one that embeds sensibly in $\mathbb{R}^d$. A Pokémon team is discrete, and any reasonable embedding would be very high dimensional. BO looked less appealing.

First I needed a battle simulator.

### Getting Battles Off the Ground

The standard Python route is `poke-env`, which connects to a Pokémon Showdown server. That brings along chat rooms, animations, and player accounts when all I wanted was a function from two teams to a winner.

I tried writing my own engine. A few weeks in, I had hit too many edge cases and had no good way to verify the engine matched the game. I found an existing engine that was exactly what I wanted, but it was written in Zig and I couldn't get it to build.

So I went back to `poke-env`, but kept the engine modular. If I get another engine working, it should slot in without touching the optimization code.

### Formalizing the Problem

To keep the search space tractable, I restricted to **Generation 1**: no held items, abilities, effort values, or natures. A team is just six species and four legal moves per species.

It is still enormous. Generation 1 has 151 species, so choosing 6 gives

$$\binom{151}{6} \approx 1.49 \times 10^{10}$$

species combinations. Choosing 4 moves from an average learnset of about 30 gives $\binom{30}{4} = 27405$ movesets per species. Altogether,

$$\binom{151}{6} \cdot \binom{30}{4}^6 \approx 6.3 \times 10^{36}$$

distinct teams. Suppose every grain of sand on Earth (about $10^{19}$) were a computer evaluating one team per second. It would take about $2 \times 10^{10}$ years, more than four times the age of the Earth.

The objective is the expected win-rate of a team $t$ against the metagame $\mathcal{M}$, the distribution of opposing teams:

$$f(t) = \mathbb{E}[\text{win-rate} \mid t, \mathcal{M}],$$

approximated by the empirical win-rate over a finite number of simulated battles. This objective is:

- **Black-box:** no closed form, only simulation.
- **Stochastic:** the same matchup can go either way.
- **Combinatorial:** teams are discrete, so gradients are meaningless.
- **Metagame-dependent:** strength is relative to opponents, not absolute.

### Choosing an Optimization Strategy

Gradient-based methods are out. Local search is suspect: swapping a single Pokémon can swing performance through matchups and synergy, so the landscape isn't smooth in any useful sense.

That points to *population-based stochastic optimization*: maintain a set of candidates and improve them iteratively without strong assumptions about the objective. I built the framework around two routines:

1. `evaluate_teams`: given a population, score each team.
2. `produce_next_generation`: given a scored population, produce the next.

This abstraction was the most useful decision in the project. Random Search and Genetic Algorithms are just two implementations of `produce_next_generation`.

Both score teams with ELO ratings from battles within the population. ELO captures relative performance (beating a strong team earns more than beating a weak one), so the population builds a ranking without a fixed set of opponents.

#### Random Search

Keep the highest-rated teams and fill the rest with teams sampled uniformly from the legal space. It is nearly memoryless; the only continuity is the retained top performers.

#### Genetic Algorithm

Build new teams from the top performers. Crossover combines two parents into a child that inherits some Pokémon from each. Mutation randomly replaces a Pokémon or alters a move with some fixed probability. This biases exploration toward regions that have already shown promise.

#### Comparing the Two

ELO is relative to the population, so ELO scores from different runs aren't comparable. A team that dominates a weak population can have high ELO and still lose badly to a strong one.

This showed up in the results. The best Random Search teams had higher ELO than the best Genetic Algorithm teams, but that was ELO inflation: Random Search generates many terrible teams, which become free ELO for mediocre ones.

For a fair comparison, I evaluated the final teams against a fixed gauntlet of 30 human-made OU teams and computed empirical win-rates.

### The Agent Problem

A battle depends on the agents playing the teams, not just on the teams. The objective is really

$$f(t) = \mathbb{E}[\text{win-rate} \mid t, \pi, \mathcal{M}],$$

where $\pi$ is the agent policy. We are finding the best team *as played by a particular agent*.

I used `poke-env`'s `SimpleHeuristicPlayer`, a small step up from random play. A stronger agent would have been a project of its own, but this is a real limitation: the agent ignores type matchups, switching, prediction, and long-term planning. The resulting teams aren't competitive in any human sense. They are robust under naive play.

The Dream Eater story is representative, not an outlier. Optimization exploits any bias in the evaluation pipeline. If the agent has blind spots, the optimizer will find teams that exploit them rather than teams that generalize to stronger play.

### Results

I ran both methods for 10 generations. That is not many; hundreds or thousands would be better. ELO over time:

<p align="center">
  <img src="/blog/assets/pokemon-team-opt/aggregate_performance.png" 
  alt="Aggregate ELO vs generation" style="max-width: 100%;">
</p>

And win-rates against the gauntlet:

<p align="center">
  <img src="/blog/assets/pokemon-team-opt/aggregate_evaluation.png" 
  alt="Mean win rate vs gauntlet" style="max-width: 100%;">
</p>

Neither method does well in absolute terms, but the Genetic Algorithm has a slight edge in win-rate. The ELO inflation is visible in the first plot: Random Search accumulates higher ELO despite performing worse on the gauntlet.

The GIFs below show the highest-ELO team at each generation. I didn't track the cumulative best, because ELO changes meaning as the population changes.

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

The win-rates are modest. The framework is the better measure of the project.

Most of the work went into the evaluation pipeline (battle simulation, orchestration, bookkeeping), not the optimizers. Once the `evaluate_teams` and `produce_next_generation` abstraction was clean, adding an algorithm took almost no time.

The bigger lesson is what optimization does: it finds teams that score well under your specific evaluation setup. If that setup has biases, such as a weak agent or a skewed metagame sample, the optimizer will find and exploit them. That is not a failure of the method. It is the method working as intended. Making the evaluation reflect what you actually care about is the designer's job.

### What Comes Next

The obvious next step is a better battle agent: a stronger heuristic or a learned policy. Co-evolving teams and agents would be more interesting, since better play puts pressure on teams to improve.

On the algorithm side: more search strategies, direct gauntlet win-rate instead of ELO, or combinatorial BO. The code is built around Generation 1. Extending it to later generations is conceivable but would need a significant refactor.

For now this is a proof of concept. Team building fits cleanly as noisy black-box combinatorial optimization, and the framework makes it cheap to try new methods. These 10-generation results don't show that any method works well, only that the pipeline does.

Thanks for reading!

### Code

The full code is on [GitHub](https://github.com/nathan-cantafio/pokemon).
