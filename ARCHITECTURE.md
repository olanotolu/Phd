# TLM Architecture Specification v0.1

## Name

**Thermodynamic Language Model (TLM)** — a physics-inspired language model that represents text as a discrete energy landscape and generates structure via thermodynamic sampling instead of next-token prediction.

---

## Core Idea

TLM does **not** predict the next token like a Transformer.

Instead, it:

1. Assigns an **energy** to a whole sequence of tokens.
2. Learns parameters so that **realistic sequences have low energy** and unrealistic sequences have higher energy.
3. Uses **Gibbs / block sampling** to move from random noise toward structured, low-energy configurations.

It is a **language model as a physical system**: text = states, factors = interactions, generation = energy minimization.

---

## High-Level Architecture

```
                ┌───────────────────────────┐
                │        Input Text         │
                │   "yhwojxab njyma..."     │
                └────────────┬──────────────┘
                             │  (tokenize chars)
                             ▼
                ┌───────────────────────────┐
                │    Symbol Encoder         │
                │  chars → indices (0..K)   │
                └────────────┬──────────────┘
                             │  (discrete states)
                             ▼
        ┌────────────────────────────────────────────┐
        │         TLM Energy Graph (Factor Graph)    │
        │                                            │
        │   Nodes: one per position (x₁..xₙ)         │
        │   Factors: local interactions (bigrams,     │
        │            trigrams, position terms, etc.) │
        │                                            │
        │   E(x) = Σ_factor ψ_factor(x_neighbors)    │
        └────────────┬───────────────────────────────┘
                     │  (defines energy landscape)
                     │
         Gibbs / Block Sampling Loop
                     │
        ┌────────────▼───────────────────────────────┐
        │        Thermodynamic Sampler (Gibbs)       │
        │  • repeatedly resample variables in blocks│
        │  • accept moves that lower energy          │
        │  • settles into low-energy configurations  │
        └────────────┬───────────────────────────────┘
                     │  (sampled states)
                     ▼
                ┌───────────────────────────┐
                │  Sampled States → Text     │
                │ indices → characters       │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌───────────────────────────┐
                │   Generated Sequence       │
                │  structured, low-energy    │
                │  text-like patterns        │
                └───────────────────────────┘
```

Training happens by adjusting the **factor weights** so that:

- real sequences = low energy
- mismatched / random sequences = higher energy

---

## Components (v0.1)

### 1. Symbol Encoder

- **Input**: raw text (character-level in v0.1)
- **Output**: integer indices in `[0, K)` where K = alphabet size
- **Current alphabet**: `a–z`, space, comma, period (29 characters)

### 2. Nodes (Discrete Variables)

- One node per position in the sequence: `x₁, x₂, …, xₙ`
- Each node takes one of K discrete values (characters)

### 3. Energy Graph (Factor Graph)

A set of **factors** connects neighboring nodes.

**v0.1 uses bigram factors**:

- each pair `(xᵢ, xᵢ₊₁)` has a K×K interaction matrix Wᵢ

Optional unary factors (bias per character / position) can be added.

**Total energy**:

```
E(x) = Σᵢ ψ_bigram(xᵢ, xᵢ₊₁) + Σᵢ ψ_unary(xᵢ)
```

**Lower energy = more "stable" / "structured" sequence.**

### 4. Thermodynamic Sampler

Uses **Gibbs sampling** or **block Gibbs**:

- resample characters at one or more positions conditioned on neighbors
- Iteratively reduces energy (like annealing) and finds **attractor states** (stable patterns).
- Sampling temperature can control randomness vs stability.

### 5. Training Objective

**v0.1**: **pseudo-likelihood** over observed sequences.

**Future versions**: contrastive divergence, score matching, or KL-gradient estimation.

**Goal**: assign lower energy to real text than to corrupted or random text.

### 6. Pattern Analyzer (Optional but included)

Computes:

- vowel/space/punctuation ratios
- bigram frequency statistics
- detection of repeating motifs

Compares random data vs TLM samples to show emergence of structure.

---

## Capabilities (v0.1)

- Learns **local structure** in text (e.g., bigrams, repeated motifs).
- Produces **stable, repeated patterns** from pure noise.
- Demonstrates **emergent order** in symbolic sequences.
- Acts as a **microscopic prototype** of a future thermodynamic language model.

---

## Limitations (Honest)

- Still character-level only.
- No long-range syntax or semantics yet.
- Not competitive with Transformers on language tasks.
- The value is conceptual + structural, not performance-focused (yet).

---

## Technical Details

### Energy Function

For a sequence `x = (x₁, x₂, ..., xₙ)`:

```
E(x) = -Σᵢ Wᵢ[xᵢ, xᵢ₊₁]
```

where `Wᵢ` is a K×K weight matrix for position `i`.

### Sampling

Block Gibbs sampling with alternating updates:

1. Update even positions given odd positions
2. Update odd positions given even positions
3. Repeat until convergence

### Training

Pseudo-likelihood objective:

```
L = -Σᵢ log P(xᵢ | x_{neighbors})
```

Gradient update:

```
∇W = E_data[bigram_counts] - E_model[bigram_counts]
```

---

## Future Extensions

- **Trigram factors**: `ψ(xᵢ, xᵢ₊₁, xᵢ₊₂)`
- **Long-range factors**: interactions between distant positions
- **Hierarchical factors**: multi-scale structure
- **Conditional generation**: given context, sample continuation
- **Real text training**: Wikipedia, code, domain-specific corpora

---

## References

- LeCun, Y., et al. "A Tutorial on Energy-Based Learning" (2006)
- Extropic THRML Framework
- Borges, J. L. "The Library of Babel" (1941)

