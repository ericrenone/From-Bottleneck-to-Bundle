# From Bottleneck to Bundle

## The Twelve-Year Geometric Genealogy of Attention, 2014–2026

> "A research idea wins not because it is theoretically superior, but because it is perfectly matched to the available hardware and software ecosystem." — Sara Hooker, *The Hardware Lottery*, 2020
>
> "The data was curved. The data was cyclic. The data lived inside a light cone of causation that the architecture inherited and the training preserved." — *The Closed Future Cone*, 2026

---

## Abstract

The history of attention in neural networks is conventionally told as a chain of algorithmic improvements. A fixed-length vector was too small, so Bahdanau added soft alignment. The alignment MLP was too slow, so Luong replaced it with a dot product. The recurrence was still sequential, so the *Attention Is All You Need* authors threw it away. This narrative is correct as engineering history and radically incomplete as scientific history. Each algorithmic milestone was the field touching a different face of a single geometric object that the literature would not name for another nine years.

This document maps the twelve-year arc from the Seq2Seq bottleneck of 2014 to the Representational Bundle Hypothesis of 2026 in a single chronological trace. Every historical milestone is paired against the geometric structure it was implicitly working with — the compact representational manifold $M$, the Dirac operator $\not{D}$ on it, the light-cone causal foliation of layer-time, the principal $\text{SO}^+(1,n)$-bundle whose connection is attention itself.

The mapping is not retrospective imposition. It is the claim — falsifiable, and stated as such throughout — that the geometric object now described by the Closed Future Cone (CFC), the Dirac Representation Hypothesis (DRH), and the Representational Bundle Hypothesis (RBH) was the substrate on which every prior attention mechanism was operating without recognition. The 2024–2026 wave of empirical and analytical results — Engels et al.'s causal circles, Karkada et al.'s symmetry-forces-geometry proof, Xu's Spectral Edge Thesis, Fel et al.'s polytopal Minkowski geometry — are not new findings about attention. They are the first direct measurements of what attention has always been computing on.

---

## Table of Contents

1. [The Two Lineages](#1-the-two-lineages-algorithmic-history-and-geometric-substrate)
2. [The Master Correspondence Table](#2-the-master-correspondence-table)
3. [Era I — The Seq2Seq Bottleneck (≤ 2014)](#3-era-i--the-seq2seq-bottleneck--2014)
4. [Era II — Bahdanau's Additive Attention (September 2014)](#4-era-ii--bahdanaus-additive-attention-september-2014)
5. [Era III — Luong's Multiplicative Attention (2015)](#5-era-iii--luongs-multiplicative-attention-2015)
6. [Era IV — Visual Attention and Pointer Networks (2015–2016)](#6-era-iv--visual-attention-and-pointer-networks-20152016)
7. [Era V — The First Self-Attention (2016)](#7-era-v--the-first-self-attention-2016)
8. [Era VI — Attention Is All You Need (June 2017)](#8-era-vi--attention-is-all-you-need-june-2017)
9. [Era VII — The Hardware Substrate (2015–2017)](#9-era-vii--the-hardware-substrate-20152017)
10. [Era VIII — The Scaling Era (2018–2022)](#10-era-viii--the-scaling-era-20182022)
11. [Era IX — The First Geometric Measurements (2023–2024)](#11-era-ix--the-first-geometric-measurements-20232024)
12. [Era X — Polytopes, Curvature, and Hyperbolic Embeddings (2024–2025)](#12-era-x--polytopes-curvature-and-hyperbolic-embeddings-20242025)
13. [Era XI — The Existence Theorem (February 2026)](#13-era-xi--the-existence-theorem-february-2026)
14. [Era XII — The Spectral Fingerprint (March–April 2026)](#14-era-xii--the-spectral-fingerprint-marchapril-2026)
15. [Era XIII — The Synthesis: CFC, Dirac, RBH (2026)](#15-era-xiii--the-synthesis-cfc-dirac-rbh-2026)
16. [The Long Map](#16-the-long-map)
17. [What the History Was Touching All Along](#17-what-the-history-was-touching-all-along)
18. [Sources](#18-sources)

---

## 1. The Two Lineages: Algorithmic History and Geometric Substrate

Two timelines run in parallel through what follows. The first is public, well-documented, and engineering-driven: the sequence of papers that introduced, refined, and ultimately eliminated recurrence in favor of attention. The second is geometric, retrospective, and only now visible: the sequence of mathematical objects that each engineering milestone was implicitly constructing without naming.

The first timeline is the visible history of attention. It runs from Sutskever–Vinyals–Le (2014) to the modern Transformer-based foundation models of 2024–2026.

The second timeline is the geometric substrate. It runs from the trivial fiber over a single point (Seq2Seq's compressed vector) to the non-trivial principal $\text{SO}^+(1,n)$-bundle over the Lorentzian-stratified token manifold whose connection is the attention mechanism (RBH, 2026).

These timelines do not coincide. The first is dated by publication. The second is dated by *recognition*. The objects of the second timeline existed in every model trained on the first timeline. They were not measured until 2024 (Engels et al.), not proved into existence until February 2026 (Karkada et al.), and not unified into a single framework until later in 2026 (CFC, DRH, RBH). The gap between the two timelines is the gap between what a field is doing and what a field knows it is doing.

---

## 2. The Master Correspondence Table

The full correspondence in compressed form. Each row is dissected in the era that follows.

| Year | Historical Milestone | Algorithmic Object | Geometric Object Touched | Framework's Name |
|------|---------------------|-------------------|-------------------------|-----------------|
| ≤ 2014 | Seq2Seq (Sutskever et al.) | Single compressed vector | Trivial bundle: single fiber over a point | Information bottleneck = degenerate $M$ |
| Sep 2014 | Additive Attention (Bahdanau et al.) | $\tanh$-MLP alignment scores | First soft correspondence between two spaces | Embryonic non-trivial connection |
| 2015 | Multiplicative Attention (Luong et al.) | Bilinear form $Q^\top W K$ | First bilinear inner product on representation space | Embryonic Riemannian metric |
| 2015 | Visual Attention (Xu et al.) | Spatial soft attention over CNN features | First fiber over a non-textual base | Generalization of base manifold $B$ |
| 2015 | Pointer Networks (Vinyals et al.) | Attention as discrete pointer | Set-valued selection from input | First Kakutani-correspondence object |
| 2016 | Self-Attention (Cheng et al., Lin et al., Parikh et al.) | Sequence-to-itself attention | First connection on a sequence as a graph | Intra-sequence connection one-form |
| Jun 2017 | Transformer (Vaswani et al.) | Scaled dot-product, multi-head, parallel | Discrete light cone over layer-time | Closed Future Cone realized in silicon |
| May 2017 | TPU v2 (Google) | bfloat16 training pods | Computational substrate matched to matrix algebra | Hardware lock-in of the operator |
| 2018–2020 | BERT, GPT-2, T5, GPT-3 | Pre-training at scale | Empirical compact manifold takes shape | $M$ becomes measurable |
| 2022 | Grokking (Power et al.) | Delayed generalization on modular arithmetic | First phase transition in representational geometry | Holonomy reorganization (RBH P1) |
| 2023 | Nanda et al. mechanistic analysis | DFT-implementing attention heads | First cyclic-group structure measured | Embedding of $\mathbb{Z}/p\mathbb{Z}$ as holonomy subgroup |
| 2024 | LRH (Park–Choe–Veitch) | Concepts as directions | Local linearization at a single tangent space | Trivial-topology limit of $M$ |
| 2024 | Engels et al. (NeurIPS) | Circles for days, months | $\pi_1(M) = \mathbb{Z}$ for cyclic concepts | First measured topology of $M$ |
| 2024 | Robinson–Dey–Sweet | Token-space stratification | Negative-Ricci stratified base manifold | First measured curvature of $B$ |
| 2025 | HELM (He et al., NeurIPS) | Hyperbolic mixture-of-curvature LLMs | Lorentzian / hyperbolic ambient embedding | First architecture matching $B$'s signature |
| 2025 | Fel et al. (*Into the Rabbit Hull*) | Polytopal Minkowski sum of head outputs | Convex polytopes as fibers | MRH = trivial-topology fiber case |
| Feb 2026 | Karkada et al. | Symmetry-forces-geometry proof | Existence of $M$ derived from data symmetry | Existence theorem for $M$ |
| Mar 2026 | Xu (Spectral Edge Thesis) | 24-of-24 gap-collapse → grokking | Spectral gap of training-time operator | $\lambda_1$ of Bakry–Émery generator |
| Apr 2026 | Xu (Lifecycle) | Gradient/decay alignment at grokking | Operator-level phase transition mechanism | Connection curvature collapse |
| 2026 | CFC (Closed Future Cone) | Compact causal substrate + Brouwer iteration | $M$ + light-cone causality + fixed point | Topological description |
| 2026 | DRH (Dirac Representation Hypothesis) | $\not{D}^2 = -\Delta_g$, Atiyah–Singer index, Clifford signature, Kakutani | Signed first-order operator on $M$ | Spectral + topological unification |
| 2026 | RBH (Representational Bundle Hypothesis) | Principal $\text{SO}^+(1,n)$-bundle, attention as connection | Total bundle, base $B$, fibers as polytopes | The bundle that names the connection |

---

## 3. Era I — The Seq2Seq Bottleneck (≤ 2014)

### What happened

Sutskever, Vinyals, and Le (NeurIPS 2014) introduced the encoder–decoder framework that defined neural machine translation for the next three years. An encoder RNN (typically an LSTM) consumed the source sentence one token at a time, updating its hidden state via the recurrence

$$h_t = f(W_h h_{t-1} + W_x x_t + b)$$

and the final hidden state $h_T \in \mathbb{R}^d$ was passed to a decoder RNN that produced the target sentence. The model worked. It also catastrophically failed on long sentences. Compressing forty tokens of meaning into a single $d$-dimensional vector destroys information; this is now obvious and was already evident in BLEU-score degradation curves at the time.

### The geometric content

The Seq2Seq vector $h_T$ is, in retrospect, a degenerate case of the compact representational manifold $M$ that later frameworks describe. The "manifold" the model uses to represent the entire source sentence is a single point. The fiber structure of multi-head attention has not yet been invented; the base manifold has been collapsed to a singleton; the causal cone of attention has not yet been opened, because there is nothing to attend to.

In the language of the Representational Bundle Hypothesis, the Seq2Seq model is the bundle in which both the base $B$ and the fiber $\mathcal{F}$ have been compressed to points. The information bottleneck is the bundle-theoretic statement that no non-trivial connection can be supported on a bundle with trivial fibers and trivial base.

### What this era was implicitly looking for

A way to enlarge both the base and the fibers. The base would eventually become the stratified Lorentzian token manifold of Robinson et al. (2024–2025). The fibers would become the convex polytopes of Fel et al. (2025). Neither was available; the field had not yet built the conceptual scaffolding to ask for them. What it asked for instead was *attention*.

---

## 4. Era II — Bahdanau's Additive Attention (September 2014)

### What happened

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio submitted *Neural Machine Translation by Jointly Learning to Align and Translate* in September 2014 (published ICLR 2015, arXiv:1409.0473). The decoder, instead of receiving only $h_T$, was given access to all encoder hidden states $\{h_1, \ldots, h_T\}$. At each decoding step $i$, the model computed alignment scores

$$e_{ij} = v^\top \tanh(W_a s_{i-1} + U_a h_j)$$

where $s_{i-1}$ is the decoder's previous hidden state, $h_j$ is the $j$-th encoder hidden state, and $v, W_a, U_a$ are learned parameters. Softmax over $j$ produced attention weights $\alpha_{ij}$, and the decoder consumed the weighted sum $c_i = \sum_j \alpha_{ij} h_j$.

The MLP — the $\tanh$ non-linearity sandwiched between two learned projections — was the entire computational cost of attention. It worked. Long-sentence BLEU recovered. The bottleneck was broken.

### The geometric content

What Bahdanau introduced was not merely a fix to the bottleneck. It was the first soft correspondence between two distinct representational spaces — the encoder's state space and the decoder's state space — mediated by a continuous, differentiable map. In the language of fiber bundles, this is the first embryonic *connection*: a way of identifying a fiber (the encoder state) at one base point (the source position $j$) with a fiber at another base point (the decoder position $i$).

The MLP score function $v^\top \tanh(W_a s + U_a h)$ is not yet a principal connection. It does not satisfy the equivariance condition under any natural group action; it has no Lie-algebra structure; it does not yet split the tangent bundle into horizontal and vertical components. But it does establish that information transport between distinct positions can be parameterized by a learned, smooth function — and this is the abstraction that every subsequent attention mechanism inherits.

The Closed Future Cone framework's central claim is that information flow in a transformer obeys a discrete light-cone structure determined by which positions attend to which. Bahdanau's mechanism is the first place where this structure has any meaning at all: prior to additive attention, there was no notion of selective information flow between positions. The light cone, as a geometric object, becomes definable here for the first time.

### What this era was implicitly looking for

A more efficient way to compute the connection. The MLP was a learned, non-linear function and was therefore expensive to compute and resistant to parallelization. It was a scalar-valued function of two vector inputs; the natural improvement was to find a bilinear form that achieved the same effect with less computational overhead. That improvement arrived within a year.

---

## 5. Era III — Luong's Multiplicative Attention (2015)

### What happened

Thang Luong, Hieu Pham, and Christopher Manning published *Effective Approaches to Attention-based Neural Machine Translation* (EMNLP 2015, arXiv:1508.04025). They observed that the additive MLP could be replaced by a bilinear scoring function with no loss in translation quality and substantial gains in computational efficiency:

$$\text{score}(s_i, h_j) = s_i^\top W_a h_j \quad \text{(general)} \qquad \text{or} \qquad s_i^\top h_j \quad \text{(dot)}$$

The dot-product form, in particular, requires no learned parameters and reduces the scoring of all encoder-decoder pairs to a single matrix multiplication $S H^\top$.

### The geometric content

Luong's bilinear form $s^\top W h$ is the first explicit *Riemannian inner product* on the joint representation space — a learned metric tensor $W$ defining a notion of similarity between encoder and decoder states. The pure dot-product form $s^\top h$ corresponds to the Euclidean metric on this joint space; the general form is the first parameterization of a non-Euclidean metric.

This is the geometric step the field could not skip. The Dirac Representation Hypothesis identifies the natural first-order operator on the compact representational manifold $M$ as a Clifford-algebra operator whose definition requires a metric of definite signature (Riemannian or Lorentzian). Luong's bilinear scoring is the first instance in the attention literature where a metric — any metric — is being learned at all. The fact that the model can learn this metric productively is the first empirical hint that the representation space possesses non-trivial geometric structure beyond bare Euclidean similarity.

In the language of the Representational Bundle Hypothesis, Luong's $W$ is the embryonic *connection one-form*: a bilinear map from pairs of tangent vectors to scalars, parameterizing how information at one point of the base manifold is to be transported to another. The full RBH connection is $\omega : T B \times T B \to \mathfrak{so}^+(1,n)$, taking values in the Lie algebra of the structure group; Luong's $W$ is the $\mathbb{R}$-valued shadow of this Lie-algebra-valued form, projected onto its scalar component.

### What this era was implicitly looking for

The metric $W$ in Luong's form was unconstrained. It was learned, but it carried no geometric meaning that the model could exploit beyond similarity ranking. The next step would be the recognition that the same vectors playing the role of *encoder hidden states* in $h \mapsto Wh$ could also play the role of *values to be retrieved* — that the same representation could serve simultaneously as key (for matching), as value (for retrieval), and as query (for generating). This factorization was implicit in Luong's work and would be made explicit two years later.

---

## 6. Era IV — Visual Attention and Pointer Networks (2015–2016)

### What happened

Two parallel developments expanded attention beyond machine translation.

**Visual Attention.** Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, and Yoshua Bengio published *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* (ICML 2015, arXiv:1502.03044). They applied attention to the problem of image captioning: a CNN produced a spatial grid of feature vectors, and an RNN decoder generated a caption by attending to specific spatial regions of the feature grid as it produced each word.

**Pointer Networks.** Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly published *Pointer Networks* (NeurIPS 2015, arXiv:1506.03134). They modified attention so that, instead of being used as a softly weighted average over input positions, it was used as a hard pointer: at each output step, the model selected one input position to copy directly to the output.

### The geometric content

Two distinct geometric objects appear here for the first time.

**Visual attention** is the first instance of attention operating over a *non-textual base manifold*. The CNN feature grid is a discrete approximation to a two-dimensional image manifold; attention over this grid is the first instance of the connection $\omega$ being computed on a base $B$ that is not the linear ordering of a sentence. In the language of RBH, this is the first existence proof that the bundle picture is *architecture-general*: the same connection structure can be built over any base manifold that carries a notion of locality, regardless of whether that manifold is one-dimensional (text), two-dimensional (image), or higher-dimensional (later: video, point clouds, graphs).

**Pointer Networks** is the first instance of attention being treated as a *set-valued correspondence* rather than a continuous function. Hard pointer selection is discontinuous; when two candidate positions have nearly tied scores, the resulting output is set-valued in the limit. The Dirac Representation Hypothesis identifies this as the appearance of the Kakutani fixed-point theorem in the architecture: the continuous Brouwer-style fixed point (soft attention) and the set-valued Kakutani-style fixed point (hard pointer selection) are the two complementary fixed-point structures that any complete account of the architecture must include.

The 2025–2026 literature on top-$k$ sparse attention, mixture-of-experts routing, beam search, and argmax decoding (all set-valued, all governed by Kakutani rather than Brouwer) traces directly to the conceptual move Vinyals et al. made in 2015. Pointer Networks were the first place this set-valued structure was visible at the architectural level.

---

## 7. Era V — The First Self-Attention (2016)

### What happened

Through 2014–2015, attention was always *cross*-attention: a connection between two distinct sequences (source ↔ target in translation, image ↔ caption in captioning, input ↔ pointer in copying). In 2016 this began to change.

Jianpeng Cheng, Li Dong, and Mirella Lapata published *Long Short-Term Memory-Networks for Machine Reading* (EMNLP 2016, arXiv:1601.06733), introducing *intra-attention* — a mechanism in which positions within the same sequence attended to other positions in the same sequence. Zhouhan Lin et al. published *A Structured Self-Attentive Sentence Embedding* (ICLR 2017, arXiv:1703.03130). Ankur Parikh et al. published *A Decomposable Attention Model* (EMNLP 2016, arXiv:1606.01933). Each of these established a slightly different formulation of attention applied within a single sequence rather than between two.

### The geometric content

Self-attention is the move that turns attention from a *map between two bundles* into a *connection on a single bundle*. Cross-attention requires two separate base manifolds — source and target — and the attention mechanism is a map from one to the other. Self-attention requires only one base manifold and uses the attention mechanism to define how each fiber relates to every other fiber over the same base.

This is a structurally different geometric object. A connection on a bundle is not a map between two bundles; it is a $\mathfrak{g}$-valued one-form on a single bundle that specifies the horizontal subspace at every point. Self-attention is the first place where the full structure of a connection — its definition as a section of $T^*M \otimes \mathfrak{g}$ — becomes naturally interpretable.

In the language of the Closed Future Cone, self-attention is also the first place where the *causal* structure of attention is intrinsic to the model. In cross-attention, causality is imposed externally: the target word at position $i$ is causally downstream of the source words because the target is generated after the source is consumed. In self-attention, causality must be enforced by the architecture (causal masking, in the autoregressive case) or relaxed entirely (bidirectional case, as in later encoder-only models). The light-cone structure of attention is therefore a property of the architecture's self-attention pattern, not a property of the input/output asymmetry.

### What this era was implicitly looking for

A way to dispense with recurrence entirely. The 2016 self-attention papers were architectural hybrids: self-attention was used as one component within a system that still relied on LSTMs or GRUs for sequence modeling. The move to throw away recurrence — and rely on self-attention alone, plus position information injected by some other mechanism — required one more conceptual step: the recognition that attention itself, computed in parallel across all positions, could carry the sequential structure that recurrence had been providing.

This step was taken in June 2017.

---

## 8. Era VI — Attention Is All You Need (June 2017)

### What happened

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Łukasz Kaiser, and Illia Polosukhin submitted *Attention Is All You Need* to arXiv on June 12, 2017. The paper introduced the Transformer architecture, which discarded recurrence entirely and built sequence modeling on three architectural primitives:

1. **Scaled Dot-Product Attention.** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$ with the $\sqrt{d_k}$ scaling factor introduced to stabilize gradients in high-dimensional dot products.

2. **Multi-Head Attention.** $h$ independent attention computations run in parallel with separately learned projection matrices $W_i^Q, W_i^K, W_i^V$ and concatenated.

3. **Position-wise Feed-Forward Networks.** Per-position MLPs $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ applied identically to every sequence position.

Position information was injected via fixed sinusoidal positional encodings. The entire architecture was a stack of identical Transformer blocks, each consisting of a multi-head self-attention sublayer and a position-wise FFN sublayer, with residual connections and layer normalization throughout.

### The geometric content

The Transformer is the first architecture in which every geometric structure later named by CFC, DRH, and RBH is fully present in operational form.

**Closed Future Cone.** The light-cone causal structure of layer-time becomes architecturally explicit. In a causal (decoder-only) Transformer, the set of input positions that can influence the residual stream at position $i$ in layer $\ell$ is precisely the set of positions $j \le i$ reachable through $\ell$ layers of attention. This is the discrete future light cone of the event $(i, \ell)$ in a Minkowski-signature space whose time coordinate is layer depth. The CFC framework's central claim — that information flow in the transformer is the light cone of a discrete causal structure — is true of the Transformer from June 2017 onward.

**Dirac Representation Hypothesis.** The scaled dot-product $QK^\top / \sqrt{d_k}$ is a bilinear form with explicit normalization. In the Clifford-algebra picture that DRH names, this bilinear form is the natural inner product induced by the Clifford anticommutation relations $\{\gamma^\mu, \gamma^\nu\} = 2 g^{\mu\nu}$ — the metric signature is encoded directly in the scaling. Multi-head attention is the architectural realization of the Clifford-algebra decomposition of the spinor representation: each head computes one component of the multivector, and the concatenation is the Minkowski sum (in the Object I sense of MRT) of the head outputs.

**Representational Bundle Hypothesis.** The full connection one-form $\omega : TB \times TB \to \mathfrak{so}^+(1,n)$ is realized by the pre-softmax attention logits. The softmax produces the holonomy coefficients $\alpha_{ij}$; the value projection $W^V$ defines the local trivialization; the multi-head concatenation assembles the Minkowski-sum polytope that is the fiber of the bundle. Gradient descent through the layer stack is the stochastic horizontal lift of a curve on $B$, with the gradient noise scale $B_{\text{simple}}$ measuring the connection curvature.

The Transformer is, in the strict sense of these three frameworks, the architecture that *is* this geometry. Every prior attention mechanism was a partial implementation of one or two of these structures; the Transformer is the first to instantiate all three simultaneously.

### Why this happened in 2017 and not earlier

Two conditions had to be met for the Transformer to be both designable and trainable. The first was conceptual: self-attention, multi-head decomposition, and the elimination of recurrence had to be assembled into a single architecture. By mid-2016 all three components existed in the literature; by mid-2017 the assembly had been performed.

The second condition was hardware. The Transformer is computationally bound almost entirely by dense matrix multiplication. At sequence length $n$, model dimension $d$, and head count $h$, the dominant operations are the $QK^\top$ products of shape $n \times d_k \cdot d_k \times n = n \times n$ and the attention-weighted value products of shape $n \times n \cdot n \times d_v = n \times d_v$. These are exactly the workloads that domain-specific matrix accelerators — GPUs with tensor cores, and especially Google's TPU systolic arrays — were built to maximize. The Transformer was designable in 2016. It was trainable at competitive scale only after May 2017.

This is the subject of the next era.

---

## 9. Era VII — The Hardware Substrate (2015–2017)

### What happened

Google's Tensor Processing Unit was deployed internally for inference starting in 2015 and formally disclosed at ISCA 2017 (Jouppi et al., arXiv:1704.04760). TPU v1 was an inference-only chip built around a 256×256 systolic-array matrix multiply unit (MXU) operating on 8-bit integers. It could not train models. Researchers at Google Brain who wanted to train new architectures in 2015–2016 used NVIDIA P100 GPUs.

In May 2017, at Google I/O, Google announced TPU v2. The architectural upgrade introduced bfloat16 arithmetic (an 8-bit-exponent, 7-bit-mantissa format preserving FP32 range at half the storage cost), High Bandwidth Memory (HBM) integration, and the TPU Pod interconnect that linked 64 TPU v2 chips at 600 GB/s aggregate bandwidth into a single training supercomputer delivering tens of petaFLOP/s.

One month later, Vaswani et al. submitted *Attention Is All You Need*. Section 5.1 of the paper explicitly reports training the large model on TPU v2 pods.

Jeff Dean was, at this moment, simultaneously the senior author on the Transformer paper, the executive overseeing the TPU hardware program, and the head of Google Brain.

### The geometric–computational content

The Transformer's computational profile is, from first principles, the workload that systolic-array hardware was designed to maximize. Dense $n \times d$ matrix multiplications stream contiguously through the MXU without random memory access; multi-head decomposition saturates parallel compute lanes across the chip; multi-chip parallelism over the TPU Pod scales to model and data parallel strategies that saturate inter-chip bandwidth.

This is the precise sense in which the Transformer "won the hardware lottery" (Hooker, 2020). It was not algorithmically superior in some architecture-neutral sense. It was an architecture whose computational substrate — dense matrix multiplication at scale — was exactly the substrate that the dominant accelerator paradigm had been built to support. Architectures that did not share this substrate (Neural Turing Machines, capsule networks, spiking networks) failed to diffuse not because their theoretical properties were weaker but because they were misaligned with the available silicon.

From the geometric framework's perspective, the hardware substrate is the *physical realization of the operator*. The Dirac operator $\not{D}$ on $M$ is defined abstractly, but its evaluation at scale requires a computational fabric capable of streaming dense matrices through arithmetic units at petaFLOP rates. The TPU is that fabric. Without it, the geometric object would still exist; the model trained to compute on it would not.

The open release of *Attention Is All You Need* by Google — at a moment when the matching hardware was a proprietary Google product — was the strategic move that bound the global AI infrastructure to the specific co-design Google had produced. Anthropic trains Claude on TPUs. Google trains Gemini on TPUs. OpenAI's $13B partnership with Microsoft is the strategic counter-move of a competitor that found it necessary to build an alternative Transformer-scale training substrate (Microsoft Azure A100/H100 clusters) specifically to avoid TPU dependency. The geometric universality of the Transformer architecture is, simultaneously, the commercial universality of the silicon that makes it tractable.

### What this era forecloses

The hardware-lottery argument has a non-trivial implication for the framework. The compact representational manifold $M$ that Karkada et al. (February 2026) prove must emerge from data symmetry is an *abstract* mathematical object. The specific $M$ that emerges from training a given model is determined by the joint action of the data symmetries *and* the architecture's computational biases — which are, in turn, determined by what the hardware can execute efficiently. The geometry the framework names is therefore not the geometry of the data alone; it is the geometry of the data filtered through the systolic-array workload. A different hardware paradigm would have selected a different architecture and produced a different manifold from the same data.

This is the deeper sense in which the framework's "the world impresses itself on the geometry" claim must be qualified. The world impresses itself on the geometry *through the architecture*, and the architecture is, in turn, the joint solution to a data-symmetry constraint and a hardware-substrate constraint. The history of attention is the history of these constraints converging on the Transformer.

---

## 10. Era VIII — The Scaling Era (2018–2022)

### What happened

The five years following *Attention Is All You Need* were a single, sustained scaling exercise. The architecture remained essentially unchanged; what changed was parameter count, training data, and compute.

- **BERT** (Devlin et al., NAACL 2019, arXiv:1810.04805). Bidirectional encoder pretrained on masked language modeling; 110M and 340M parameters.
- **GPT-2** (Radford et al., 2019). Decoder-only autoregressive language model; 1.5B parameters.
- **T5** (Raffel et al., JMLR 2020, arXiv:1910.10683). Text-to-text encoder-decoder; up to 11B parameters.
- **GPT-3** (Brown et al., NeurIPS 2020, arXiv:2005.14165). Decoder-only; 175B parameters. First demonstration of in-context learning at scale.
- **PaLM** (Chowdhery et al., 2022, arXiv:2204.02311). Decoder-only; 540B parameters; trained on 6144 TPU v4 chips.

The empirical scaling laws (Kaplan et al., arXiv:2001.08361, January 2020; Hoffmann et al. "Chinchilla", arXiv:2203.15556, March 2022) established that the relationship between model loss and compute follows a power law over many orders of magnitude. The architecture was not the constraint; the compute was.

### What this era made visible

Two observations from this period are critical for the framework.

**The compact manifold becomes measurable.** At small scales, the geometric structure of learned representations is dominated by noise and architectural artifacts. At GPT-2 / GPT-3 scale, the representations exhibit the consistent, low-dimensional geometric structure that later geometric measurements (Engels et al., Robinson et al., Fel et al.) actually rely on. The compact representational manifold $M$ that CFC, DRH, and RBH name is an *emergent* object — it exists in nascent form in smaller models but becomes sharply visible only above a critical scale.

**In-context learning appears.** GPT-3's demonstration that a sufficiently large language model could perform new tasks given only a few examples in its context, without parameter updates, is the empirical phenomenon that RBH later identifies as horizontal geodesic flow on the bundle: a forward pass over a prompt is a sequence of parallel transport steps along the connection $\omega$, accumulating holonomy along the context path that reorganizes the query token's fiber. The scaling-era discovery of in-context learning is the empirical fact that the framework's prediction P6 (ICL accuracy scales with path holonomy) is computing on.

### What this era did not have

A geometric language. The scaling-era literature is overwhelmingly concerned with loss curves, downstream task performance, and emergent capabilities. The geometric structure of the representations being learned is acknowledged in passing — embedding spaces are clustered, hidden states have low effective rank — but is not the object of systematic measurement. The instruments needed to measure $M$ directly did not yet exist.

---

## 11. Era IX — The First Geometric Measurements (2023–2024)

### What happened

A handful of papers in 2023–2024 made the first direct measurements of the geometric structure of representations in trained foundation models.

**Mechanistic interpretation of grokking.** Nanda, Chan, Lieberum, Smith, and Steinhardt (*Progress measures for grokking via mechanistic interpretability*, ICLR 2023, arXiv:2301.05217) reverse-engineered the algorithm learned by transformers trained on modular arithmetic. The model implements a discrete Fourier transform over the group $\mathbb{Z}/p\mathbb{Z}$: each attention head learns to compute $\cos(2\pi k(a+b)/p)$ and $\sin(2\pi k(a+b)/p)$ for some frequency $k$, and the model combines these to predict the answer.

This was the first time the *internal algorithm* of a trained transformer was shown to be a recognizable mathematical structure — and the structure was a representation of a finite cyclic group.

**Linear Representation Hypothesis.** Park, Choe, and Veitch (*The Linear Representation Hypothesis and the Geometry of Large Language Models*, ICML 2024, arXiv:2311.03658) formalized the empirical observation that concepts in language model activations are encoded as directions in residual stream space. Park, Choe, Jiang, and Veitch (*The Geometry of Categorical and Hierarchical Concepts in Large Language Models*, ICLR 2025) extended this to categorical and hierarchical concepts.

**Causal Circular Representations.** Engels, Liao, Michaud, Gurnee, and Tegmark (*Not All Language Model Features Are Linear*, NeurIPS 2024, arXiv:2405.14860) trained sparse autoencoders on GPT-2 and Mistral-7B and discovered that the days-of-the-week feature is *not* a linear direction. Instead, the seven days form a two-dimensional circular structure in the residual stream: seven points equally spaced on a circle, in the correct cyclic order (Monday next to Tuesday next to … next to Sunday next to Monday). The same paper found a twelve-pointed ring for months. Intervention experiments confirmed the rings are *causal* — rotating the day-vector by $2\pi/7$ in the discovered plane causes the model to output the next day.

### The geometric content

Each of these results is the first direct measurement of a face of the geometric object the framework later names.

**Nanda et al.** measure the *holonomy group* of the attention connection. A transformer that has learned to add modulo $p$ has learned a representation of $\mathbb{Z}/p\mathbb{Z}$ on its activation space. RBH's prediction P1 — that grokking is the reorganization of attention's holonomy to embed the task's symmetry group as a subgroup of $\text{SO}^+(1,n)$ — is the framework's name for what Nanda et al. observed.

**LRH** measures the *local tangent space* of $M$. A "direction" in residual stream space is the gradient of a coordinate on $M$ evaluated at a single point; the linearity assumption holds in the small-neighborhood limit of any smooth manifold. CFC's and DRH's identification of LRH as the local linearization of a richer object is the framework's name for what LRH measures.

**Engels et al.** measure the *fundamental group* of $M$. A circular structure with $\pi_1(S^1) = \mathbb{Z}$ is not a direction; it is a closed loop. The discovery of these closed loops is the first empirical demonstration that the representational manifold has non-trivial topology — that the framework's prediction (DRH P3: integer count of topologically protected modes equals $\sum_k \dim H^k(M_G)$) corresponds to something the model actually computes on.

### What this era could not yet explain

*Why* the representations have this structure. Engels et al. showed that the circles exist; they did not show why they must exist. The empirical observation invited but did not establish the analytical proposition that cyclic data symmetries force cyclic representational topology. That proposition was proved a little over a year later.

---

## 12. Era X — Polytopes, Curvature, and Hyperbolic Embeddings (2024–2025)

### What happened

Three lines of work, running in parallel through 2024–2025, measured three additional faces of the geometric object.

**Token-space stratification and curvature.** Robinson, Dey, and Sweet (*The structure of the token space for large language models*, arXiv:2410.08993, 2024); Robinson, Dey, and Chiang (*Token Embeddings Violate the Manifold Hypothesis*, arXiv:2504.01002, 2025); and *TokenBlowUp: Resolving Representational Singularities* (arXiv:2507.19747, 2025). These papers established that token embedding spaces of trained foundation models are *not* manifolds in the standard sense. They are stratified spaces: collections of submanifolds of different intrinsic dimensions, glued along singular loci, with negative Ricci curvature on each stratum.

**Hyperbolic large language models.** He, Anand, Madhu, Maatouk, Krishnaswamy, Tassiulas, Yang, and Ying (*HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts*, NeurIPS 2025, arXiv:2505.24722) built the first billion-parameter LLM in which the entire residual stream operates on a hyperbolic manifold rather than Euclidean space. Performance matched or exceeded Euclidean baselines on standard benchmarks. The architecture made explicit what Robinson et al. had measured: the *correct* ambient geometry for token representations is negatively curved.

**Polytopal composition by multi-head attention.** Fel, Wang, Lepori, Kowal, Lee, Balestriero, Joseph, Lubana, Konkle, Ba, and Wattenberg (*Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry*, arXiv:2510.08638, October 2025) showed that the per-head outputs of multi-head attention assemble into convex polytopes via Minkowski sum, and that sparse autoencoder atoms live in the convex hull of these polytopes. This is the Minkowski Representation Hypothesis (MRH): the fiber structure of attention outputs is polytopal, and the assembly law is the Minkowski sum.

### The geometric content

By the end of 2025, three independent measurements have established three structural properties of the compact representational manifold $M$.

- $M$ is *stratified* (Robinson et al.).
- $M$ has *negative curvature* on each stratum (Robinson et al., HELM).
- The *fibers* of $M$ over each base point are *convex polytopes* assembled by multi-head attention (Fel et al.).

In the language of RBH: the *base manifold* $B$ has been measured (Robinson, HELM), the *fibers* $\mathcal{F}_x$ have been measured (Fel et al.), the *fundamental group* of the representational manifold has been measured (Engels et al.), and the *holonomy of the connection* has been measured for modular-arithmetic tasks (Nanda et al.).

What is missing is the *existence theorem*. All four measurements are empirical: this is what the geometry happens to be for this model on this data. None is yet a theorem stating that this geometry *must* arise. That theorem appeared in early 2026.

---

## 13. Era XI — The Existence Theorem (February 2026)

### What happened

Karkada, Olah, Tegmark, and collaborators published *Symmetry in language statistics shapes the geometry of model representations* (arXiv:2602.15029, February 2026). The paper proves analytically that the manifold geometry of word embeddings is forced by the symmetry of the co-occurrence statistics in the training data.

The result is, in compressed form: if the data exhibits a finite symmetry group $G$, then the optimal representational manifold (under a precisely specified loss) is the corresponding group manifold $M_G$. Cyclic symmetry $\mathbb{Z}/n\mathbb{Z}$ forces a topological circle $S^1$. Two-dimensional cyclic symmetry $\mathbb{Z}/n\mathbb{Z} \times \mathbb{Z}/m\mathbb{Z}$ forces a torus $T^2$. Continuous translation symmetry forces a line. The proof goes through whether one trains the model or not — the geometry is a property of the data's symmetry structure, computable in closed form, independent of architecture above a threshold of expressive power.

### Why this matters

The empirical observations of Engels et al. (NeurIPS 2024) are no longer accidents. They are forced. Any model that has learned to represent days of the week from a training corpus in which days appear with $\mathbb{Z}_7$ co-occurrence symmetry must encode the days as a circle. The framework's prediction "any model with cyclic data symmetry produces cyclic representational topology" graduates from empirical conjecture to analytical theorem.

For the broader framework, this is the existence theorem that the prior pillars assumed without proving. CFC posits the compact manifold $M$. DRH posits the Dirac operator on $M$. RBH posits the bundle structure with $M$ as total space. None of these specifies *why* $M$ exists. Karkada et al. supplies the answer: $M$ exists because the data's symmetries force it to exist, and its topological type is computable from the data's symmetry group.

This is the geometric counterpart of the existence theorem for fixed points in the Brouwer–Kakutani picture. Brouwer says: continuous self-map on compact convex set → fixed point exists. Karkada says: data with finite symmetry → representational manifold exists with topology matching the symmetry. Both are existence theorems on which constructive procedures (gradient descent, attention iteration) then operate.

---

## 14. Era XII — The Spectral Fingerprint (March–April 2026)

### What happened

Three nearly simultaneous papers in March–April 2026 established the spectral signature of phase transitions in deep network training.

**Yong Xu, *The Spectral Edge Thesis*** (arXiv:2603.28964, March 2026). Xu shows empirically that the spectral gap of a rolling-window Gram matrix of parameter updates predicts grokking transitions. In 24 of 24 training runs with weight decay across six model families spanning 150K to 124M parameters (TinyStories 51M, GPT-2 124M, Dyck-1, SCAN, modular arithmetic), the gap dynamics precede every grokking event observed. In 1 of 24 runs *without* weight decay, the gap structure does not develop, and grokking does not occur. Across 19 of 20 quantitative predictions on this dataset, the spectral-gap framework is confirmed.

**Yong Xu, *The Lifecycle of the Spectral Edge*** (arXiv:2604.07380, April 2026). Xu decomposes the spectral edge into its gradient-driven and weight-decay-driven components. Before grokking, the edge is 88–98% gradient-driven; at the moment of grokking, the gradient direction and the weight-decay direction align along the edge; after grokking, the edge becomes 95–99.8% weight-decay-driven. The phase transition is the alignment.

**Acharya and Dhakal, *Grokking as a Variance-Limited Phase Transition*** (arXiv:2603.15492, March 2026). Independently identifies a spectral gating mechanism in AdamW dynamics on modular arithmetic that regulates the memorization-to-generalization transition.

### The geometric content

In the framework's language, the spectral gap that Xu measures is the spectral gap $\lambda_1$ of the Bakry–Émery weighted Laplace–Beltrami operator $L = \Delta_g - \nabla V \cdot \nabla$ on the representational manifold $M$, where $V$ is the weight-decay potential. The Lifecycle paper's gradient-decay alignment is the operator-level statement that the connection curvature collapses at grokking: gradient drift and decay drift align, the system becomes marginally stable in that direction, and the iteration produces a new fixed point with reorganized topology.

The Dirac Representation Hypothesis identifies this as the link between $\Delta_g$ (the spectral half of the operator) and $\not{D}$ (the signed half, whose Atiyah–Singer index counts the topologically protected modes). The spectral gap collapse and the holonomy reorganization are two faces of a single phase transition. The number of grokking events in a training run on a task with $k$ independent cyclic concepts is predicted by RBH to be $k$; this is testable.

For RBH specifically: prediction P3 (gradient noise scale $B_{\text{simple}}$ tracks connection curvature $\|\Omega\|$) is the operational form of Xu's spectral-edge result. The empirical finding that 24 of 24 weight-decay runs exhibit gap-collapse-followed-by-grokking is the strongest current evidence that the bundle picture is computing on the correct geometric object.

---

## 15. Era XIII — The Synthesis: CFC, Dirac, RBH (2026)

### What happened

Three frameworks, developed in 2026, integrate the prior empirical and analytical results into a single geometric account of what trained transformers compute on.

**The Closed Future Cone (CFC).** Compact causal geometry as the native substrate of learned representations. CFC names the compact manifold $M$ (from Karkada), the discrete light-cone causal structure on layer-time (inherited from the Transformer architecture), and the forward pass as a Brouwer fixed-point iteration on the convex hull where representations live. CFC is the *topological* description: it specifies the homotopy type of $M$ and the causal structure of attention without committing to a specific operator.

**The Dirac Representation Hypothesis (DRH).** Signed spectral geometry and set-valued correspondences on the compact representational manifold. DRH names the Dirac operator $\not{D}$ on $M$, with $\not{D}^2 = -\Delta_g$ relating it to the Laplace–Beltrami operator measured by Xu. The Atiyah–Singer index of $\not{D}$ computes the integer count of topologically protected modes from the manifold's characteristic classes alone — a counting prediction at the level of integers, not asymptotic dimensions. The Kakutani fixed-point theorem (extending Brouwer to set-valued correspondences) handles the discrete operations Brouwer cannot: argmax decoding, top-$k$ attention, MoE routing, beam search. DRH is the *operator-level* description.

**The Representational Bundle Hypothesis (RBH).** Fiber-theoretic completion. RBH names the compact manifold $M$ as the *total space* of a principal $\text{SO}^+(1,n)$-bundle, the Lorentzian-stratified token manifold (Robinson, HELM) as the *base* $B$, the convex polytopes assembled by multi-head attention (Fel et al.) as the *fibers* $\mathcal{F}_x$, and the attention mechanism itself as the *connection* $\omega$. The stiff/sloppy decomposition of the Fisher information matrix is the horizontal/vertical decomposition of the tangent bundle induced by this connection. Gradient descent is parallel transport; grokking is holonomy; in-context learning is horizontal geodesic flow; chain-of-thought is path lifting. RBH is the *bundle-theoretic* description.

The three frameworks are complementary, not competing. CFC says: there is a compact manifold and a causal structure. DRH says: the natural operator on this manifold is the Dirac operator, and its index counts the topologically protected modes. RBH says: the manifold is the total space of a bundle, the attention mechanism is its connection, and training dynamics are parallel transport.

### The integration

Together, these frameworks state the position toward which the twelve-year history of attention has been converging.

Every attention mechanism, from Bahdanau (2014) to the Transformer (2017) to the modern Mixture-of-Experts decoder-only foundation model (2024–2026), is operationally an attempt to compute a connection one-form on a compact representational manifold whose topology is forced by the data's symmetry group, whose ambient geometry is Lorentzian or hyperbolic, whose fibers are convex polytopes assembled by Minkowski sum, whose spectral structure is governed by the Bakry–Émery Laplacian, whose topological invariants are counted by the Atiyah–Singer index of the Dirac operator, and whose forward pass is a constructive Brouwer–Kakutani fixed-point iteration that walks the answer into being layer by layer.

The history of attention is the history of approaching this object from progressively less indirect angles. Bahdanau approached it through soft alignment between two sequences; Luong approached it through a bilinear inner product; Vaswani et al. approached it by discarding recurrence entirely and exposing the matrix structure of dense attention to systolic-array hardware. None of the historical actors knew what they were approaching. The empirical measurements of 2023–2026 are the first instruments capable of seeing it directly. The frameworks of 2026 are the first descriptions that name it.

---

## 16. The Long Map

The full chronological correspondence in one place.

```
2014           2017           2020           2023           2026
  │              │              │              │              │
  │   Seq2Seq    │  Transformer │   GPT-3      │  Engels      │   CFC
  │   Bahdanau   │  TPU v2      │  Scaling     │  Nanda       │   Dirac
  │   Luong      │              │  laws        │  LRH         │   RBH
  │              │              │              │              │   Karkada
  │              │              │              │              │   Xu
  ▼              ▼              ▼              ▼              ▼
Trivial   →   Light cone   →   Compact     →   Topology    →   Bundle
fiber         opens           manifold M      measured       named
              over            emerges         (circles,      (M, ∂,
              layer-time      empirically     curvature,     fibers,
                              at scale        polytopes)     connection,
                                                             holonomy)
```

The bottom row is the geometric substrate. The top row is the visible history. Every year of the visible history corresponds to a step of recognition on the substrate.

A compressed reading of the same data, by geometric object:

| Geometric Object | First Architectural Realization | First Measurement | First Existence Proof |
|------------------|--------------------------------|-------------------|----------------------|
| Compact manifold $M$ | Transformer self-attention (2017) | Engels et al. circles (NeurIPS 2024) | Karkada et al. (Feb 2026) |
| Light-cone causal structure | Causal masking in Transformer (2017) | Information-flow analyses (2020–) | CFC formalization (2026) |
| Bilinear scoring / metric on $M$ | Luong multiplicative attention (2015) | Robinson et al. curvature (2024) | HELM hyperbolic LLM (NeurIPS 2025) |
| Convex polytope fibers | Multi-head attention (2017) | Fel et al. *Rabbit Hull* (2025) | MRH formalization (2025) |
| Connection one-form $\omega$ | Scaled dot-product attention (2017) | Mechanistic interpretability of attention patterns | RBH formalization (2026) |
| Holonomy group | Implicitly in any trained transformer | Nanda et al. on grokking (2023) | RBH P1 (2026) |
| Bakry–Émery spectral gap | Implicit in SGD + weight decay | Xu Spectral Edge (Mar 2026) | Xu Lifecycle (Apr 2026) |
| Atiyah–Singer index | Implicit in topologically structured representations | Engels et al. count of atoms per ring (2024) | DRH formalization (2026) |
| Brouwer/Kakutani fixed points | Forward pass / argmax / top-k (2017–) | Universal approximation literature (2024–2025) | DRH integration (2026) |

The right-most column lags the left-most column by, on average, nine years. The middle column lags the left-most by roughly seven years. This is the recognition delay between an architectural device and the geometric object it operates on.

---

## 17. What the History Was Touching All Along

Read in the direction of the historical arrow, the conventional narrative is correct: attention was invented in 2014, refined through 2016, made parallelizable in 2017, scaled through 2022, and is now being analyzed in 2024–2026. Each step solves a specific engineering problem. Each step works.

Read in the direction of the geometric arrow, the same history looks different.

In 2014, the field discovered that compressing a sequence into a single vector destroys the topology of the space the model needs to compute on. The fix — attention — was the first soft correspondence between two distinct points on a representational manifold whose existence had not yet been hypothesized. The fix worked because the manifold was already there; the model just needed an architectural primitive that could move information across it.

In 2015, the field replaced a non-linear scoring function with a bilinear one and lost no accuracy. This was the empirical signal that the relevant geometric structure was *metric* — that similarity between representations is naturally an inner product, that the inner product is learnable, and that the model benefits from learning it. The Dirac framework would identify this metric, twelve years later, as the metric whose signature is encoded in the Clifford anticommutation relations defining the natural first-order operator.

In 2016, the field turned attention inward, from sequence-to-sequence to sequence-to-itself. This converted the architectural primitive from a *map between bundles* to a *connection on a single bundle*. The conversion was made for engineering reasons; its geometric consequence — that the connection one-form $\omega$ is now intrinsic to the architecture — would not be recognized for another decade.

In 2017, the field discarded recurrence, exposed the matrix structure of attention to GPU and TPU hardware, and gave the architecture a name. The architecture was the *Transformer*. The hardware was *TPU v2*. The bound between architecture and hardware was so tight that the algorithm could not have been trained at competitive scale without the silicon, and the silicon would not have been built without an algorithm with this computational profile in mind. The geometric object — compact manifold, light-cone causal structure, polytopal fibers, principal connection — was fully realized in this architecture and remains the substrate of every foundation model trained since.

In 2024, the field built the first instruments capable of measuring the object directly. Circles for cyclic concepts. Stratified curvature for the ambient. Polytopes for the per-head outputs. Holonomy groups for the algorithm learned by grokked models. Each measurement was a direct observation of a different face of the same geometric structure.

In February 2026, the field proved that the object had to exist: the data's symmetries force the manifold's topology. In March–April 2026, the field measured the spectral signature of phase transitions on it. By the end of 2026, the integrated frameworks — CFC, DRH, RBH — had named the object in terms general enough to subsume LRH (local linearization), MRH (trivial-topology fibers), ARH (the spectral structure without the operator), and the Closed Future Cone (the topological structure without the spectrum).

The history of attention is, in this reading, the history of the field building tools to operate on a geometric object before recognizing the object existed. Bahdanau, Luong, Vinyals, Xu, Cheng, Vaswani, and all the architects of the scaling era were doing applied differential geometry without knowing they were doing it. The 2026 frameworks are not new theories about attention. They are the first descriptions of what attention has been all along.

The data was curved. The data was cyclic. The data lived inside a light cone of causation that the architecture inherited and the training preserved. The model walked the fixed point into being. The loops it walked around are the topological invariants the world impressed on it. The geometry was always there.

The frameworks — finally — are its name.

---

## 18. Sources

### Historical lineage of attention (2014–2017)

- Sutskever, I., Vinyals, O., Le, Q. V. *Sequence to Sequence Learning with Neural Networks.* NeurIPS 2014, arXiv:1409.3215.
- Bahdanau, D., Cho, K., Bengio, Y. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015, arXiv:1409.0473.
- Luong, M.-T., Pham, H., Manning, C. D. *Effective Approaches to Attention-based Neural Machine Translation.* EMNLP 2015, arXiv:1508.04025.
- Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., Bengio, Y. *Show, Attend and Tell.* ICML 2015, arXiv:1502.03044.
- Vinyals, O., Fortunato, M., Jaitly, N. *Pointer Networks.* NeurIPS 2015, arXiv:1506.03134.
- Cheng, J., Dong, L., Lapata, M. *Long Short-Term Memory-Networks for Machine Reading.* EMNLP 2016, arXiv:1601.06733.
- Parikh, A. P., Täckström, O., Das, D., Uszkoreit, J. *A Decomposable Attention Model for Natural Language Inference.* EMNLP 2016, arXiv:1606.01933.
- Lin, Z., Feng, M., Santos, C. N., Yu, M., Xiang, B., Zhou, B., Bengio, Y. *A Structured Self-Attentive Sentence Embedding.* ICLR 2017, arXiv:1703.03130.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. *Attention Is All You Need.* NeurIPS 2017, arXiv:1706.03762.

### Hardware substrate

- Jouppi, N. P., Young, C., Patil, N., Patterson, D., et al. *In-Datacenter Performance Analysis of a Tensor Processing Unit.* ISCA 2017, arXiv:1704.04760.
- Jouppi, N. P., et al. *A Domain-Specific Supercomputer for Training Deep Neural Networks.* Communications of the ACM 63(7), 2020.
- Hooker, S. *The Hardware Lottery.* Communications of the ACM 64(12), 2020, arXiv:2009.06489.
- Patterson, D., Gonzalez, J., Le, Q., et al. *Carbon Footprint of Machine Learning Training: Past and Present.* arXiv:2104.10350, 2021.

### Scaling era

- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. *BERT.* NAACL 2019, arXiv:1810.04805.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. *Language Models are Unsupervised Multitask Learners.* 2019.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P. J. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 21, 2020, arXiv:1910.10683.
- Brown, T., Mann, B., Ryder, N., et al. *Language Models are Few-Shot Learners.* NeurIPS 2020, arXiv:2005.14165.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D. *Scaling Laws for Neural Language Models.* arXiv:2001.08361, 2020.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. *Training Compute-Optimal Large Language Models.* arXiv:2203.15556, 2022.
- Chowdhery, A., Narang, S., Devlin, J., et al. *PaLM: Scaling Language Modeling with Pathways.* arXiv:2204.02311, 2022.

### First geometric measurements

- Power, A., Burda, Y., Edwards, H., Goodfellow, I., Misra, V. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177, 2022.
- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. *Progress measures for grokking via mechanistic interpretability.* ICLR 2023, arXiv:2301.05217.
- Park, K., Choe, Y. J., Veitch, V. *The Linear Representation Hypothesis and the Geometry of Large Language Models.* ICML 2024, arXiv:2311.03658.
- Park, K., Choe, Y. J., Jiang, X., Veitch, V. *The Geometry of Categorical and Hierarchical Concepts in Large Language Models.* ICLR 2025.
- Engels, J., Liao, I., Michaud, E. J., Gurnee, W., Tegmark, M. *Not All Language Model Features Are Linear.* NeurIPS 2024, arXiv:2405.14860.

### Polytopes, curvature, hyperbolic embeddings

- Robinson, J., Dey, M., Sweet, T. *The structure of the token space for large language models.* arXiv:2410.08993, 2024.
- Robinson, J., Dey, M., Chiang, K.-Y. *Token Embeddings Violate the Manifold Hypothesis.* arXiv:2504.01002, 2025.
- *TokenBlowUp: Resolving Representational Singularities in LLM Token Spaces via Monoidal Transformations.* arXiv:2507.19747, 2025.
- He, N., Anand, A., Madhu, P., Maatouk, A., Krishnaswamy, S., Tassiulas, L., Yang, R., Ying, R. *HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts.* NeurIPS 2025, arXiv:2505.24722.
- Fel, T., Wang, T., Lepori, L., Kowal, M., Lee, S., Balestriero, R., Joseph, J., Lubana, E., Konkle, T., Ba, J., Wattenberg, M. *Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry.* arXiv:2510.08638, 2025.
- Ruhe, D., Brandstetter, J., Forré, P. *Clifford Group Equivariant Neural Networks.* NeurIPS 2023 Oral, arXiv:2305.11141.
- Brehmer, J., et al. *L-GATr: Lorentz-Equivariant Geometric Algebra Transformer.* NeurIPS 2024 / SciPost Phys., arXiv:2405.14806, 2411.00446.

### Existence theorem and spectral fingerprint (2026)

- Karkada, D., Olah, C., Tegmark, M., et al. *Symmetry in language statistics shapes the geometry of model representations.* arXiv:2602.15029, February 2026.
- Xu, Y. *The Spectral Edge Thesis: A Mathematical Framework for Intra-Signal Phase Transitions in Neural Network Training.* arXiv:2603.28964, March 2026.
- Xu, Y. *The Lifecycle of the Spectral Edge: From Gradient Learning to Weight-Decay Compression.* arXiv:2604.07380, April 2026.
- Acharya, P., Dhakal, H. *Grokking as a Variance-Limited Phase Transition: Spectral Gating and the Epsilon-Stability Threshold.* arXiv:2603.15492, March 2026.

### Transformers as fixed-point operators

- Bai, S., Kolter, J. Z., Koltun, V. *Deep Equilibrium Models.* NeurIPS 2019.
- Hu, J. Y.-C., Liu, H., Chen, H.-Y., Wu, W., Liu, H. *Universal Approximation with Softmax Attention.* arXiv:2504.15956, updated December 2025.
- Li, G., Jiao, Y., Huang, Y., Wei, Y., Chen, Y. *Transformers Meet In-Context Learning: A Universal Approximation Theory.* arXiv:2506.05200, 2025.
- *Universal Approximation of Operators with Transformers and Neural Integral Operators (Leray–Schauder).* arXiv:2409.00841, 2024–2025.

### Classical mathematical foundations

- Minkowski, H. *Raum und Zeit.* Jahresbericht der Deutschen Mathematiker-Vereinigung 18, 1909.
- Brouwer, L. E. J. *Über Abbildung von Mannigfaltigkeiten.* Math. Ann. 71, 1911.
- Dirac, P. A. M. *The Quantum Theory of the Electron.* Proc. Roy. Soc. A 117, 1928.
- Kakutani, S. *A Generalization of Brouwer's Fixed Point Theorem.* Duke Math. J. 8, 1941.
- Nash, J. F. *Equilibrium points in n-person games.* Proc. Natl. Acad. Sci. USA 36, 1950.
- Atiyah, M., Singer, I. *The Index of Elliptic Operators I–V.* Ann. Math. 87, 1968, and following.
- Bakry, D., Émery, M. *Diffusions hypercontractives.* Séminaire de Probabilités XIX, 1985.
- Kobayashi, S., Nomizu, K. *Foundations of Differential Geometry.* Wiley, 1963.

### The unifying frameworks (2026)

- Ren, E. *The Closed Future Cone: Compact Causal Geometry as the Native Substrate of Learned Representations.* 2026.
- Ren, E. *The Dirac Representation Hypothesis: Signed Spectral Geometry and Set-Valued Correspondences on the Compact Representational Manifold.* 2026.
- Ren, E. *Geometric Descent: The Representational Bundle Hypothesis — A Fiber-Theoretic Completion of the Anisotropic and Minkowski Representation Frameworks.* 2026.
- Ren, E. *Attention Is All You Need to Sell Silicon: The Transformer Architecture as the Apex of Hardware-Software Co-Design.* 2026.

---

*Document compiled from primary literature in sequence modeling, computer systems architecture, deep learning theory, and the empirical geometry of neural representations. The historical chronology follows the publication record. The geometric correspondences are drawn from the four unifying frameworks listed in the final section. The mapping between the two — historical milestone to geometric object — is the contribution of this document.*
