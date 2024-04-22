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


| Title                                                                                         | ref | Category | Comment |
| --------------------------------------------------------------------------------------------- | --- | -------- | ------- |
| [Black Box Adversarial Prompting for Foundation Models](https://arxiv.org/pdf/2302.04237.pdf) |     |          |
|                                                                                               |

#### Jailbreak

- [AutoDAN](https://openreview.net/pdf?id=7Jwpw4qKkb) Can we develop an approach that can automatically generate stealthy
jailbreak prompts?
- [Jailbreaking Attack against Multimodal Large Language Model](https://arxiv.org/pdf/2402.02309.pdf) Strong modeltransferability reveal a connection between MLLM-jailbreaks and LLM-jailbreaks
- [ArtPrompt](https://arxiv.org/pdf/2402.11753.pdf) LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs.



| Title                                                                                         | ref | Category | Comment                                                                                |
| --------------------------------------------------------------------------------------------- | --- | -------- | -------------------------------------------------------------------------------------- |
| [Black Box Adversarial Prompting for Foundation Models](https://arxiv.org/pdf/2302.04237.pdf) |     |          |
| [AutoDAN](https://openreview.net/pdf?id=7Jwpw4qKkb)                                           |     |          | Can we develop an approach that can automatically generate stealthy jailbreak prompts? |
|                                                                                               |

### Defense Strategy (optional)
- spelling check is a unique defense method for textual data as spelling of words/ grammar of sentence are fixed in practical and can be easily detected
- Uses gradient-based token optimization to enforce
harmless outputs. [Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2401.17263)

### Evaluation Metric
- [AttackEval](https://arxiv.org/abs/2401.09002)