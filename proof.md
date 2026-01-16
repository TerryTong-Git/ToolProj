# Blackwell Garbling Argument for Arm 1 ≤ Arm 2

## 1. Theoretical Setup (Analogy to Bayesian Inference in In-Context Learning)

Large language models (LLMs) trained with next-token prediction often exhibit behaviors that are well-described as *implicit Bayesian inference*: conditioned on a prompt, the model appears to infer latent structure that explains the prompt and then predicts outputs by marginalizing over that latent structure. This viewpoint has been used to explain in-context learning as posterior inference under a latent-variable pretraining distribution.

In this work, we apply an analogous perspective to **algorithmic reasoning**. We view an LLM’s intermediate reasoning trace as an internal representation that compresses task-relevant information from an input problem instance. Different representation regimes (e.g., code-like traces vs. natural-language traces) can be compared as *different observation channels* of the same underlying task. Our goal is to formalize when one representation regime is provably at least as informative as another for downstream task solving, independent of the particular decoder used.

Concretely, we compare:

- **Arm 2 (Code):** an intermediate trace \(Z_{\mathrm{code}}\) that is code-like (e.g., executable or near-executable program structure).
- **Arm 1 (NL):** an intermediate trace \(Z_{\mathrm{nl}}\) that is natural-language reasoning.

We focus on the question:

> When does observing a code-like trace dominate observing an NL trace for predicting the correct answer?

We show that if NL traces can be obtained by *post-processing* code traces through a fixed translation mechanism, then code traces are **Blackwell-more-informative** than NL traces, implying weakly lower Bayes risk for any downstream decision problem (in particular, lower or equal Bayes error).

---

## 2. Task

Let:
- \(X\) be the underlying algorithmic task instance (problem statement plus any latent instance variables relevant to the correct output),
- \(Y^*(X)\) be the correct answer for instance \(X\).

The model produces an intermediate representation and then a final prediction.

We compare two representation regimes:

- **Arm 2:** observe \(Z_{\mathrm{code}}\),
- **Arm 1:** observe \(Z_{\mathrm{nl}}\).

Our target claim (Arm ordering) is:
\[
P^*_{\mathrm{code}} \;\le\; P^*_{\mathrm{nl}},
\]
where \(P^*_{\mathrm{code}}\) and \(P^*_{\mathrm{nl}}\) denote Bayes-optimal 0–1 error when predicting \(Y^*(X)\) from the corresponding representation.

---

## 3. Modeling Assumptions

### 3.1 Representation Pipeline

We assume the following causal/Markov structure is a useful abstraction of the reasoning pipeline:
\[
X \;\to\; Z_{\mathrm{code}} \;\to\; Z_{\mathrm{nl}} \;\to\; \gamma \;\to\; Y.
\]

Here:
- \(Z_{\mathrm{code}}\) is produced from \(X\),
- \(Z_{\mathrm{nl}}\) is produced from \(Z_{\mathrm{code}}\) via an LLM translation step,
- \(\gamma\) denotes any downstream latent variables inferred from the representation (e.g., an inferred algorithm),
- \(Y\) is the model’s output.

**For the Blackwell comparison, only the prefix**
\[
X \;\to\; Z_{\mathrm{code}} \;\to\; Z_{\mathrm{nl}}
\]
is required. Downstream components (\(\gamma \to Y\)) are not needed for the dominance result because they can be absorbed into the downstream decision problem and loss.

---

### 3.2 Garbling (Translation) Assumption

We model the step “translate code trace to NL trace via a fixed LLM translator” as a **garbling** operation.

**Assumption (Garbling via LLM translation).**  
There exists a stochastic kernel \(T(z_{\mathrm{nl}}\mid z_{\mathrm{code}})\), independent of \(X\), such that for all \(x\),
\[
p(z_{\mathrm{nl}} \mid x)
\;=\;
\int T(z_{\mathrm{nl}} \mid z_{\mathrm{code}})
\, p(z_{\mathrm{code}} \mid x)
\, dz_{\mathrm{code}}.
\]
Equivalently, \(X \to Z_{\mathrm{code}} \to Z_{\mathrm{nl}}\) is a Markov chain.

Intuitively, this says that the NL trace can be generated from the code trace without direct access to the original instance \(X\). Thus, relative to \(X\), \(Z_{\mathrm{nl}}\) is a (potentially noisier) post-processing of \(Z_{\mathrm{code}}\).

**Remark (Matching “native” NL traces).**  
If one also defines a “native” NL trace \(Z_{\mathrm{nl}}^{\mathrm{native}}\sim p(\cdot\mid x)\) produced directly from \(X\), then the above assumption can be interpreted as asserting that the translated NL trace \(\tilde Z_{\mathrm{nl}}:=T(Z_{\mathrm{code}})\) is distributionally close to \(Z_{\mathrm{nl}}^{\mathrm{native}}\) conditional on \(X\). The exact Blackwell result below requires only the existence of a kernel \(T\) yielding the relevant conditional \(p(z_{\mathrm{nl}}\mid x)\); approximate variants can be stated if translation is only approximately matching.

---

## 4. Decision Problem and Bayes Risk

Let \(\mathcal{A}\) be an action space (e.g., candidate answers). Let \(\ell(a,x)\) be any loss function.

A (possibly randomized) decision rule based on observation \(Z\) is a conditional distribution \(\delta(a\mid z)\).

The Bayes risk for a representation \(Z\) is
\[
\mathcal{R}^*(Z)
\;=\;
\inf_{\delta}
\;\mathbb{E}[\ell(A,X)],
\quad
A \sim \delta(\cdot \mid Z).
\]

For task solving, use 0–1 loss:
\[
\ell(a,x) = \mathbf{1}\{a \neq Y^*(x)\},
\]
in which case \(\mathcal{R}^*(Z)\) is the Bayes error, denoted \(P^*(Z)\).

---

## 5. Theorem: Blackwell Dominance of Code over NL (Arm 1 ≤ Arm 2)

**Theorem (Blackwell dominance under garbling).**  
Under the garbling assumption \(X \to Z_{\mathrm{code}} \to Z_{\mathrm{nl}}\), for any prior over \(X\) and any loss function \(\ell\),
\[
\mathcal{R}^*_{\mathrm{code}} \;\le\; \mathcal{R}^*_{\mathrm{nl}}.
\]
In particular, for 0–1 loss,
\[
P^*_{\mathrm{code}} \;\le\; P^*_{\mathrm{nl}}.
\]

This establishes **Arm 1 ≤ Arm 2**.

---

## 6. Proof (Blackwell Garbling Argument)

Fix any decision rule \(\delta_{\mathrm{nl}}(a \mid z_{\mathrm{nl}})\) operating on \(Z_{\mathrm{nl}}\).

Construct a decision rule for \(Z_{\mathrm{code}}\) by first translating and then acting:
\[
\delta_{\mathrm{code}}(a \mid z_{\mathrm{code}})
\;:=\;
\int \delta_{\mathrm{nl}}(a \mid z_{\mathrm{nl}})
\, T(z_{\mathrm{nl}} \mid z_{\mathrm{code}})
\, dz_{\mathrm{nl}}.
\]
This is a valid kernel (a convex combination of valid kernels).

Now compute the induced conditional action distribution given \(X=x\):
\[
\begin{aligned}
p_{\mathrm{code}}(a \mid x)
&=
\int \delta_{\mathrm{code}}(a \mid z_{\mathrm{code}})
\, p(z_{\mathrm{code}} \mid x)
\, dz_{\mathrm{code}} \\
&=
\int
\left[
\int \delta_{\mathrm{nl}}(a \mid z_{\mathrm{nl}})
\, T(z_{\mathrm{nl}} \mid z_{\mathrm{code}})
\, dz_{\mathrm{nl}}
\right]
p(z_{\mathrm{code}} \mid x)
\, dz_{\mathrm{code}} \\
&=
\int \delta_{\mathrm{nl}}(a \mid z_{\mathrm{nl}})
\left[
\int T(z_{\mathrm{nl}} \mid z_{\mathrm{code}})
p(z_{\mathrm{code}} \mid x)
\, dz_{\mathrm{code}}
\right]
dz_{\mathrm{nl}} \\
&=
\int \delta_{\mathrm{nl}}(a \mid z_{\mathrm{nl}})
\, p(z_{\mathrm{nl}} \mid x)
\, dz_{\mathrm{nl}} \\
&=
p_{\mathrm{nl}}(a \mid x),
\end{aligned}
\]
where the penultimate equality uses the garbling identity
\(
p(z_{\mathrm{nl}}\mid x)=\int T(z_{\mathrm{nl}}\mid z_{\mathrm{code}})\,p(z_{\mathrm{code}}\mid x)\,dz_{\mathrm{code}}.
\)

Thus, the joint distribution \(p(a,x)\) induced by:
- Arm NL with decision rule \(\delta_{\mathrm{nl}}\), and
- Arm Code with decision rule \(\delta_{\mathrm{code}}\)

is identical. Consequently,
\[
\mathbb{E}_{\mathrm{code}}[\ell(A,X)]
=
\mathbb{E}_{\mathrm{nl}}[\ell(A,X)].
\]

Since this holds for **any** \(\delta_{\mathrm{nl}}\), the optimal Bayes risk under the code representation cannot be larger:
\[
\mathcal{R}^*_{\mathrm{code}}
\;\le\;
\mathcal{R}^*_{\mathrm{nl}}.
\]
This completes the proof.

---

## 7. Interpretation

The theorem formalizes that if NL traces are obtainable by a fixed translation (garbling) of code traces, then the code trace contains weakly more task-relevant information about \(X\) than the NL trace. Any procedure that makes decisions from NL traces can be simulated from code traces by translating code to NL and then applying the same procedure. Therefore, code traces are Blackwell-more-informative, yielding weakly lower Bayes risk and, in particular, weakly lower Bayes error.
