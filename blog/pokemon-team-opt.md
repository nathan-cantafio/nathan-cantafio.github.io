---
title: Pokémon Team Building as an Optimization Problem
date: 2025-12-20
description: A population-based optimization framework for Pokémon team building, using simulated battles to evaluate noisy black-box objectives over large combinatorial spaces.
---

### Introduction

The reason I initially started this was because I needed a project idea for the statistical computation course I was taking. It was important to me that I pick a topic which had the right balance of feasibility and scope. I also wanted it to be novel in some sense.

With this in the back of my mind, I came accross a YouTube video titled something like "Pokémon as a Machine Learning Problem" (which seems to have since been taken down). It essentially made the case that competitive Pokémon is an under-explored domain for machine learning methods. There are two main aspects to competitive Pokémon, team building and the battles themselves, both of which are interesting machine learning problems. Team building requires a way of evaluating teams in an enormous search space (something on the order of $10^\{200\}$), and selecting the best possible team. Measuring "best" is also imperfect; it probably depends on the potential opponents (or as players call it, the metagame. And to me, the most obvious way to evaluate a team is just to run as many battles against as many opponents as possible and calculate a win-rate. But to complicate things further, battles are stochastic, there is some noise (so maybe multiple battles should be run per opponent? I actually don't think this is a good idea: one battle per opponent and more opponents is better). 

So to recap: team building is an example of a *noisy black-box objective function*. A noisy black-box objective function?! My immediate thought was to use Bayesian Optimization (BO). Although after working on the problem BO is tough to make work  because the search space is combinatorial. Nonetheless my interest in Pokémon and background in BO got me inspired to start the project.

### How to simulate Pokémon battles?

This was the first question I had to answer. The most common way to do so (if using Python) as far as I can tell is through the `poke-env` interface which connects to a Pokémon Showdown server. A big disadvantage of this is that it requires all the overhead of a Showdown server (the chat room, battle animations, player "accounts", etc.) and not just a battle engine which takes in two teams, simulates the battle, and returns who won. 

Because of this and some other dissatisfactions I had with `poke-env`, I actually tried writing my own battle engine. After getting started on this for a few days, I decided that there are so many things that I would need to get right, and it would be so hard to tell if my engine was "cartridge accurate" that I would be better off seeing if such an engine already existed. And it did! Unfortunately, it was coded in Zig, and I could not for the life of me get it to build. So I decided to cut my losses and go back to use `poke-env`. But I wanted to make the project as extensible as possible. Not tied to a particular engine. So that  if I ever got another engine working, it would be easy to plug it into the code and build some teams.

### Problem formulation

Now that we know how battles are simulated, the next step is to formalize team building as an optimization problem. A Pokémon team consists of six Pokémon, where each Pokémon is defined by its species and a set of four moves. In modern versions of the game (generations as they're called), additional attributes such as held items, abilities, effort values (EVs), and natures all greatly increase the dimensionality of the problem. To make the search space tractable, I restricted the problem to **Generation 1**, where Pokémon have no held items, abilities, EVs, or natures. Under these constraints, a team is fully specified by:

 - The choice of six species
 - The choice of four legal moves per species

Even under this simplification, the resulting search space is enormous (see below). 

Let $\mathcal{T}$ denote the space of all legal teams under a given competitive tier (e.g. OU - which stands for "Over Used" and is just a rule set for what's allowable). Each element $t\in \mathcal{T}$ is a valid six-Pokémon team. We wish to find a team that maximizes expected performance in battle.

To quantify performance, I define the objective function

$$f(t)=\mathbb{E}[\text{win-rate}|t, \mathcal{M}],$$

where $\mathcal{M}$ represents opposing teams/strategies drawn from the metagame. In practice this expectation is approximated by simulating a finite number of battles and computing the empirical win-rate.

This formulation introduces several challenges

 1. *Black-box objective:* there is no closed-form expression for $f(t)$. The only way to evaluate a team is to simulate battles and observe outcomes.
 2. *Stochasticity:* Pokémon battles involve randomness (critical hits, damage rolls, accuracy checks, secondary effects), so repeated evaluations of the same team could yield different results.
 3. *Combinatorial structure:* Teams are discrete objects with constraints, meaning we can't even approximate gradients
 4. *Dependence on the metagame:* A team's strength is not absolute as it depends on the distribution of opposing teams/strategies used for evaluation

Taken together, we see that Pokémon team building is an example of *noisy black-box combinatorial optimization problem*. 

### Search space size

As a quick aside I want to do a back of the envelope calculation for the size of the search space.

In generation 1 there are 151 Pokémon species. Ignoring tier restrictions for the moment, a team consists of choosing 6 distinct species from this set. Since the team order doesn't matter (to a point, the first member of the party is the lead, though a good player will use a different lead for different battles), the number of possible species combinations is 

$$\binom{151}{6}\approx 1.488 \cdot 10^9$$

For each chosen species, we must also select a moveset of 4 moves from that Pokémon's legal learnset. Learnsets vary by size quite a bit, but a conservative estimate of the average is 30 legal moves per Pokémon. This gives 

$$\binom{30}{4}=27405$$

possible movesets per Pokémon.

Putting this together, the total number of distinct teams is roughly

$$\binom{151}{6}\cdot\binom{30}{4}^6=\left(1.488\cdot 10^9\right)\cdot\left(2.7\cdot 10^4\right)^6\approx 5.4\cdot 10^\{35\}.$$

This esimate is rough, but the conclusion is legitimate. The search space is astronomically large. Even if we had computers that could evaluate one team per second, and we had as many computers as there are galaxies in the unviverse, and we had been running these computers since the dawn of time, we still wouldn't have exhausted the search space!

### Optimization Methods

Given the formulation above, the optimization problem has several properties that restrict the types of methods that are appropriate. The objective function is expensive to evaluate, noisy, and defined over an immense combinatorial space. These considerations immediately rule out many standard optimization techniques. Gradient-based methods aren't applicable as the domain is discrete. And any sort of local search method is suspect since small changes to a team (e.g. swapping a single Pokémon) could lead to big swings in performance due to matchup effects and team synergy.

Bayesian Optimization (BO) was my first idea. It's a natural choice for noisy black-box global optimization. However BO struggles on this problem for two reasons. First of all, BO generally assumes the search space is continuous, or at least can be embedded in $\mathbb{R}^d$ in some reasonable way. They also generally work best in lower dimension. A Pokémon team is not continuous, and any embedding will most likely need to be quite high dimensional. Perhaps a BO type algorithm specifically designed for combinatorial optimization would do okay (this is something I want to try).

This motivates the use of *population-based stochastic optimization methods*, which maintain and iteratively improve a set of candidate solutions rather than a single point estimate. Such methods are well suited to large discrete search spaces and make minimal assumptions about the structure of the objective. The way that I've structured my project, such a method requires implementing two routines:

 1. `evaluate_teams`: Given a population of teams, assign each team a score
 2. `produce_next_generation`: Given a scored population, generate a new population of teams, typically favouring higher performing teams.

This abstraction makes it easy to swap out optimization strategies while keeping the evaluation pipeline fixed. In particular both Random Search and Genetic Algorithms can be viewed as special cases that differ only in how these two routines are implemented.

#### Random Search

In Random Search (RS), each iteration maintains a small set of top-performing teams from the previous generation and fills the remainder of the population with teams sampled uniformly at random from the legal search space. The `evaluate_teams` routine assigns scores using an ELO rating system computed from battles within the population. ELO provides a relative measure of team strength: wins against strong teams increase a team’s rating more than wins against weaker teams, and losses against strong teams are penalized less than losses against weak teams.

The `produce_next_generation` routine then discards most of the population, retaining only the highest-rated teams and replacing the rest with newly sampled random teams. This can be viewed as an elitist variant of random search. A fully memoryless version, where the entire population is resampled each generation, would also be reasonable.


#### Genetic Algorithms

In Genetic Algorithms (GA), the `evaluate_teams` routine is identical to that used in Random Search. The difference lies entirely in `produce_next_generation`. Instead of filling the population with random teams, GA generates new teams by applying crossover and mutation operations to the top-performing teams from the previous generation.

Crossover operations combine elements from two parent teams, for example, a child team might inherit three Pokémon from one parent and three from another, with movesets copied intact. While mutation operations randomly modify a team by replacing an entire Pokémon or altering one of a Pokémon’s moves with some predefined probability. These operators preserve team legality while allowing the algorithm to explore the search space in a structured way, biasing exploration toward regions that have already shown strong performance.

#### Comparing two methods that use ELO in `evaluate_teams`

Because ELO provides only a relative measure of performance within a population, final ELO scores produced by different optimization runs or different methods are not directly comparable. As a result, a separate validation step is required to legitimately compare teams.

For final evaluation, selected teams are evaluated against a fixed set of opponent teams $\mathcal{M}$, and their empirical win-rates are computed. These win-rates serve as the final metric used to compare optimization methods. While it would be possible to evaluate the true objective directly within `evaluate_teams`, doing so would be more expensive. And I like the fact that ELO allows these methods to build teams from the ground up, without relying on a pre-defined set of opponents.

### The battle agent

Up to this point, I have implicitly assumed that given two teams, we can meaningfully evaluate their relative strength by simulating battles between them. However, a Pokémon battle is not determined by team composition alone; it also depends critically on the battle agents that control each team and decide which actions to take during play.

Formally, the objective function $f(t)$ should be interpreted as depending not only on the team $t$, but also on the policy used to play that team. 

$$f(t)=\mathbb{E}[\text{win-rate}|t,\pi,\mathcal{M}],$$

where $\pi$ is the battle agent policy used.

In other words, we are not optimizing “the best team in isolation,” but rather “the best team as played by a particular agent.” By fixing the battle agent across all simulations, we remove an important confounding factor: differences in observed performance can be attributed primarily to team composition rather than differences in decision-making quality.

For this project, I used `poke-env`’s built-in `SimpleHeuristicPlayer`. After observing a number of its battles, it became clear that this agent is far from strong. Nevertheless, I chose it because it represents a small step up from purely random play, and implementing a more sophisticated agent would have required a significant additional time investment. My primary focus was on the optimization loop itself rather than on agent design.

It is also worth emphasizing that the goal of this project was not to discover optimal play, but to identify teams that perform well on average under a fixed, consistent policy. From this perspective, the agent acts as part of the evaluation environment rather than as an object of optimization.

That said, the poor quality of the `SimpleHeuristicPlayer` does place a clear limitation on the results. A weak agent fails to exploit many strategic aspects of Pokémon battles, such as type matchups, switching, prediction, and long-term planning. As a result, the teams produced by the optimization process should not be interpreted as competitive in a human sense. Instead, they are teams that are robust under naive play.

To give a concrete example of the agent’s limitations, I observed a battle in which it repeatedly used the move Dream Eater, which fails unless the target is asleep, against an awake opponent for 24 consecutive turns, until the move ran out of uses. Only then did it switch to Psychic, which knocked out the opponent in two hits (the opponent, for its part, spent the entire battle repeatedly using Agility). While extreme, this example highlights how brittle the agent’s decision-making can be, and why the results of this project should be interpreted with appropriate caution.

### Results

Here we see the plot of ELO score against generation for both methods.

<p align="center">
	<img src="/blog/assets/pokemon-team-opt/aggregate_performance.png" 
	alt ="Aggregate ELO vs generation"
	style="max-width: 75%; height: auto;">
</p>

Here is a plot of win-rate against a strong pre-defined gauntlet of opponent teams (there are 30 human-made OU teams in the gauntlet). We see that with just 10 generations neither method does particularly well, but the genetic algorithm does have a slight edge. In contrast the best Random Search teams had higher ELO than the best Genetic Algorithm team. This is an example of why ELO cannot be compared between populations. The bad teams found by random search are really bad and end up being "free ELO" for the strong (really mediocre) teams (this is called ELO-inflation in some communities).

<p align="center">
	<img src="/blog/assets/pokemon-team-opt/aggregate_evaluation.png" alt ="Mean win rate vs gauntlet">
</p>

Here I have made GIFs of how the team evolved over time. These GIFs show the team which has highest ELO for that generation at each generation. I hummed and hawed about whether showing the team with highest yet seen ELO at each generation (in other words the highest cumulative ELO) would be better, but decided against it. ELO is a measure of relative performance. I think that if the population under which ELO is measured is always changing (which it is here), then a cumulative examination of it is somewhat meaningless (nevertheless there are cumulate plots of ELO above...)

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <img src="/blog/assets/pokemon-team-opt/team_evolution_EloGeneticAlgorithm.gif"
       alt="Team evolution (Genetic Algorithm)"
       style="max-width: 50%; height: auto;">

  <img src="/blog/assets/pokemon-team-opt/team_evolution_EloRandomSearch.gif"
       alt="Team evolution (Random Search)"
       style="max-width: 50%; height: auto;">
</div>

After just 10 generations, neither method produces teams that reliably beat the gauntlet, but the genetic algorithm shows slightly more consistent progress, while random search benefits from ELO inflation rather than genuinely stronger teams.

### Lessons learned

The biggest lesson from this project is that problem formulation matters more than the optimization algorithm. Pokémon team building is easy to describe informally, but making it precise enough to optimize revealed many hidden assumptions: how teams are evaluated, what opponents they face, and how battles are played all strongly influence what “optimal” means.

A closely related takeaway is that the evaluation loop dominates everything. Most of the engineering effort went into battle simulation, orchestration, and bookkeeping rather than into the optimization algorithms themselves. Once a clean abstraction was in place (`evaluate_teams` and `produce_next_generation`) it became trivial to swap out search strategies. This suggests that, for problems like this, investing in a flexible evaluation pipeline pays off far more than prematurely optimizing the optimizer.

Another lesson is that relative metrics are both powerful and dangerous. Using ELO inside the optimization loop was extremely convenient and computationally efficient, but it also made direct comparisons across runs impossible without a separate validation phase. This reinforced the importance of clearly separating training-time signals from final evaluation metrics.

Finally, this project made it very clear that agent quality is a bottleneck. Even a modestly weak battle agent can distort the optimization landscape, rewarding teams that exploit the agent’s blind spots rather than teams that would perform well under stronger play. While this does not invalidate the results, it significantly constrains how they should be interpreted. More broadly, this project reinforced that optimization aggressively exploits any bias in the evaluation pipeline, whether it comes from the agent, the simulator, or the scoring metric.

### Future work

There are several natural extensions to this project.

The most obvious step is to replace the battle agent. Using a stronger heuristic agent, or even a learned policy, would likely push the found teams more towards "optimal under optimal play". A vague, but interesting idea is co-evolution: jointly optimizing teams and agents. This would be highly challenging but cool to see.

Another obvious next step is to implement more algorithms beyond just RS and GA. Or change the `evaluate_teams` routine to use an opponent gauntlet rather than ELO. 

Finally, although I designed the code to work with generation 1, a natural thing to think about is extending the framework to later generations. Albeit, the code would require a rather large refactor.

Overall, this project represents a first step rather than a final solution. The results are best viewed as a demonstration that Pokémon team building can be framed cleanly as a noisy black-box combinatorial optimization problem, and that even simple stochastic methods can make meaningful progress when the evaluation pipeline is carefully designed.

Thanks for reading!

### Code availability

The full code base for this project is publicly available on
[GitHub](https://github.com/cantafionathan/pokemon)


