## Title: Survey on Adversarial Attacks to LLM

### Introduction
Commercial deployed set constraint to solve the problem 
- [PaLM 2 Technical Report](https://arxiv.org/abs/2305.10403)


### Backgrounds
#### LLM
#### Prompt Engineering
#### Adversarial Attacks
- Adversarial Space: we are not able to provide model a "perfect" dataset that covers all kinds of features we expected. [Machine Learning and Security](https://www.oreilly.com/library/view/machine-learning-and/9781491979891/)

#### Jailbreak Prompts
- [Don't Listen To Me: Understanding and Exploring Jailbreak Prompts of Large Language Models](https://arxiv.org/abs/2403.17336)
  - _What are the underlying strategies of existing jailbreak prompts and their effectiveness?_ 5 categories
  - _What is the process for humans to develop and execute semantically meaningful jailbreak attacks in the real world?_ jailbreak patterns and approaches
  - _Can humans and AI work collaboratively to automate the generation of semantically meaningful jailbreak prompts?_ software fuzzing testing
- **Jailbreak Formulation** The goal of jailbreak attacks is to prompt the model to produce output starting with specific words,  our goal is to optimize the jailbreak
prompts $J_i$ to influence the input tokens and thereby maximize the probability$$P(r_{m+1},r_{m+2},\dots,r_{m+k}|t_1,t_2,\dots,t_m)=\prod_{j=1}^kP(r_{m+j}|t_1,t_2,\dots,t_m,r_{m+1},\dots,r_{m+j})$$ [AutoDAN](https://openreview.net/pdf?id=7Jwpw4qKkb)

### White-Box Attacks


### Black-Box Attacks


#### Adversarial Attack

#### Jailbreak


### Adversarial Traing



##### Categories
- access to information
  - white-box
  - black-box
- attacking phase
  - prompt engineering
  - data manipulation
- red-team attack
  - human-in-loop
  - model-based
- attack sequence
  - jailbreak
  - misleading - performance 
  - triggers - find common weakness of model
- attacking component
  - input sentence
  - prompt
  - models

##### Weak points
- order of sentence
- ASCII art
- variations in templates
- specific examples - typos, substituting words, paraphrasing

#### Details
- [Universal Adversarial Perturbations](https://www.mendeley.com/reference-manager/reader-v2/055834a3-74f3-3228-9d6d-dc1d0bd31b51/424550e8-253a-9035-798b-0e1c448f1e66) The very begining concept of universaral attack on language model. 
  - Formulation a classic targeted/untargeted adversarial problem.
  - use cos distance to project trigger to the nearest embedding
- [AutoDAN](https://openreview.net/pdf?id=7Jwpw4qKkb) Can we develop an approach that can automatically generate stealthy
jailbreak prompts?
- [Jailbreaking Attack against Multimodal Large Language Model](https://arxiv.org/pdf/2402.02309.pdf) Strong modeltransferability reveal a connection between MLLM-jailbreaks and LLM-jailbreaks
- [ArtPrompt](https://arxiv.org/pdf/2402.11753.pdf) LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs.
- [Black Box Adversarial Prompting for Foundation Models](https://arxiv.org/pdf/2302.04237.pdf) Generate prompts, which can be standalone or prepended to
benign prompts, induce specific behaviors into the generative process under black-box setting.
  - Token Space Projection (TSP), aiming the challenge from large and discrete input token space by projecting prompts into a new embedding space.
  - bridges the continuous word embedding space with the discrete token space
- [Black-Box Prompt Learning for Pre-trained-model](https://arxiv.org/pdf/2302.04237) The model tries to create a black-box method for finding independent prompt that can be added to input sentence, _similar to the concept of triggers (UAT)_. The format is a concatenate of prompt $T$ and input sentence $S$.
  - As the gradient from predictive model can not be access directly, they use policy gradient estimator to calculate loss
  - they use pointwise mutual information and ngram structure to construct vocabulary candidate
- [TrojLLM](https://www.mendeley.com/reference-manager/reader-v2/4d2a4514-80fd-3bfb-8ffb-f0b0c9dddd67/4896ba10-6501-fa99-6ac7-c72830ded736) The method targets on the searching for prompt injection called "troj". troj prompt stays high accuracy for clean input and esay to be attacked by triggered input. The targeted prediction occurs when triggers introduced to troj.
  - Targeting on the pre-trained model. 
  - use reinforcement learning framework to find the troj
- [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://www.mendeley.com/reference-manager/reader-v2/18a6b7ab-a937-3f01-a2d4-421662ce20d8/53461633-46a2-d3e1-1230-83c0eeb69fc6) 
- [Real-World Indirect Prompt Injection](https://ui.adsabs.harvard.edu/abs/2023arXiv230212173G/abstract) Author find that by injecting prompts into data likely to be retrieved, we can affect other's systems.

|Title|Ref|Accessable Information|Attacking Format|Adversarial Component|
|---|---|---|---|---|
|Universal Adversarial Perturbations||White-box|Token Manipulation|Input Text|


| Title                                                                                         | ref | Category | Comment                                                                                |
| --------------------------------------------------------------------------------------------- | --- | -------- | -------------------------------------------------------------------------------------- |
| [Black Box Adversarial Prompting for Foundation Models](https://arxiv.org/pdf/2302.04237.pdf) |     |          | Prompts standalone or prepended to                                                     |
| benign prompts                                                                                |
| [AutoDAN](https://openreview.net/pdf?id=7Jwpw4qKkb)                                           |     |          | Can we develop an approach that can automatically generate stealthy jailbreak prompts? |
|                                                                                               |


### Defense Strategy (optional)
- spelling check is a unique defense method for textual data as spelling of words/ grammar of sentence are fixed in practical and can be easily detected
- Uses gradient-based token optimization to enforce
harmless outputs. [Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2401.17263)

### Evaluation Metric
- [AttackEval](https://arxiv.org/abs/2401.09002)


### Adversarial Example Game
- [Adversarial Example Game](https://arxiv.org/pdf/2007.00720)
  - addressing problem: 

### Code Generation: computational and cognitive
#### Robustness of code
- Preliminaries
  - a vector for site code $\mathbf{z}\in\{0,1\}^n$
  - one hot vector to select token $\mathbf{u}_i\in\{0,1\}^{|\Omega|}$, from vocabulary token space $\Omega$, for each token in $\mathcal{P}$ we have $\mathbf{u}^{n\times|\Omega|}$
  - $\mathcal{P}$ is original program, $\mathcal{P}'$ perturbed program
  - $\mathbf{1}^T\mathbf{z}\leq k$ measures the perturbation strength
- Math formulation
  - $\mathcal{P}'=(1-\mathbf{z})\cdot\mathcal{P}+\mathbf{z\cdot u}$, where $\mathbf{1}^T\mathbf{z}\leq k,\mathbf{z}\in\{0,1\}^n,\mathbf{1}^T\mathbf{u}_i=1,\mathbf{u}_i\in\{0,1\}^{|\Omega|},\forall_i$
  - $$\begin{aligned} \text{minimize}&\quad \ell_{attack}((\mathbf{1-z})\cdot\mathcal{P}+\mathbf{z\cdot u};\mathcal{P},\theta)\\ \text{subject}&\quad\mathbf{1}^T\mathbf{z}\leq k,\mathbf{z}\in\{0,1\}^n,\mathbf{1}^T\mathbf{u}_i=1,\mathbf{u}_i\in\{0,1\}^{|\Omega|},\forall_i\end{aligned}$$  