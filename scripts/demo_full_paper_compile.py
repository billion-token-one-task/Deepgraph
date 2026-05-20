#!/usr/bin/env python3
"""End-to-end: realistic full paper → 6 venue adapters → real PDF compile.

What this exercises:
  • Venue router rule-based selection (CV/NLP/ML/Theory).
  • Each adapter's ``normalize_source`` produces a venue-specific preamble.
  • Each adapter's ``copy_files`` stages the right ``.sty`` / ``.bst``.
  • Tectonic compiles every venue to a real PDF.
  • Format linter runs the 7 contract checks on the same source.

Outputs land under ``/tmp/full_paper_demo/<venue>/paper.pdf`` so reviewers
can visually inspect column-layout, header, bibstyle differences.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")
os.environ["DEEPGRAPH_DB_PATH"] = "/tmp/full_paper_demo.db"

from agents import venue_router  # noqa: E402
from agents.format_linter import lint_manuscript  # noqa: E402
from agents.manuscript_templates import get_adapter, list_adapters  # noqa: E402


PAPER_BODY = r"""\documentclass{article}
\title{Linear Attention with Positive Random Features for Long-Context Language Modelling}
\author{Anonymous}
\begin{document}
\maketitle

\begin{abstract}
We introduce a linear-time attention mechanism that retains the modelling
capacity of full softmax attention while reducing complexity from
$O(n^2 d)$ to $O(n d^2)$ for sequence length $n$ and head dimension $d$.
The key ingredient is a positive random feature (PRF) map that
unbiasedly estimates the softmax kernel while preserving causal masking
via a streaming prefix sum. On three long-context benchmarks
(PG19~\cite{rae2020pg19}, ProofPile, ZeroSCROLLS~\cite{shaham2023zeroscrolls})
our $350$M-parameter decoder matches or exceeds strong dense baselines
including Performer~\cite{choromanski2021performer},
Linformer~\cite{wang2020linformer}, and Longformer~\cite{beltagy2020longformer}
while running $4.1{\times}$ faster on $16$K-token inputs and $7.8{\times}$
faster on $32$K. A controlled ablation isolates the gain to the feature
map dimension rather than to auxiliary losses or relative positional
bias, and confirms that the streaming prefix sum recovers the full
softmax quality once the feature dimension exceeds $m \geq 256$.
\end{abstract}

\section{Introduction}
\label{sec:intro}
Transformer language models~\cite{vaswani2017attention} have become the
backbone of state-of-the-art systems across natural language
understanding and generation, yet the quadratic cost of self-attention
makes them brittle on long inputs. Recent work has therefore explored
three orthogonal axes for reducing the attention budget: (i) imposing
sparsity patterns~\cite{kitaev2020reformer,zaheer2020bigbird}, (ii) low-rank
factorisation~\cite{wang2020linformer}, and (iii) kernel
approximation~\cite{katharopoulos2020transformers,choromanski2021performer}.
A parallel line of research has questioned whether attention is needed
at all and proposed state-space alternatives~\cite{gu2022s4}, while
position-encoding improvements~\cite{press2022alibi,su2024roformer}
extend the effective receptive field of unmodified dense models.

Despite this rich landscape, a head-to-head comparison reveals that
existing linear approximations either (a) lose 1.5--3 perplexity points
against the dense softmax baseline at the same parameter count, or
(b) require auxiliary mixing losses and per-layer learned positional
buckets to recover that gap. The empirical picture has motivated a
return to dense attention for production deployments such as
FlashAttention~\cite{dao2022flashattention}, which trades algorithmic
asymptotics for IO-aware kernels. We argue the gap is in fact closable
without auxiliary losses: a sufficiently expressive positive feature
map together with a streaming prefix sum recovers softmax quality at
strictly $O(n d^2)$ cost.

\paragraph{Contributions.}
\begin{itemize}
\item We propose \textbf{PRF-LinAttn}, a linear attention block whose
      feature map is the positive random features of
      Choromanski et al.~\cite{choromanski2021performer} sampled from a
      heavy-tailed distribution to reduce variance in the long-context
      regime; Section~\ref{sec:method} gives the construction.
\item We show that the causal variant admits a streaming prefix-sum
      implementation that fits a fused CUDA kernel, achieving a
      $4.1{\times}$ wallclock speedup at $16$K tokens and $7.8{\times}$
      at $32$K against an IO-optimised
      FlashAttention~\cite{dao2022flashattention} baseline
      (Section~\ref{sec:speed}).
\item Across PG19~\cite{rae2020pg19}, ProofPile, and
      ZeroSCROLLS~\cite{shaham2023zeroscrolls} we close the perplexity
      gap to softmax at feature dimension $m\geq 256$ and outperform
      Performer~\cite{choromanski2021performer},
      Linformer~\cite{wang2020linformer}, and
      Longformer~\cite{beltagy2020longformer} on all three corpora
      (Section~\ref{sec:exp_main}).
\item A controlled ablation isolates the contribution of feature
      dimension, sampling distribution, and prefix-sum precision
      (Section~\ref{sec:ablation}), and a positional-encoding sweep
      shows the gain is orthogonal to RoPE / ALiBi
      (Section~\ref{sec:pos}).
\end{itemize}

\section{Related Work}
\label{sec:related}

\paragraph{Sparse attention.}
Reformer~\cite{kitaev2020reformer} groups queries and keys by locality-sensitive
hashing, achieving $O(n \log n)$ cost at the price of an extra hashing
hyperparameter. BigBird~\cite{zaheer2020bigbird} combines random,
windowed, and global attention to provably approximate the full
softmax kernel. Longformer~\cite{beltagy2020longformer} adopts a
similar sliding-window plus task-specific global pattern. Sparse
approximations preserve numerical fidelity inside their support but
require careful pattern design; our method dispenses with any pattern
choice.

\paragraph{Low-rank factorisation.}
Linformer~\cite{wang2020linformer} projects keys and values to a fixed
low-rank subspace, reducing attention to $O(n k)$ with rank $k$. While
fast, the projection couples the model to a maximum sequence length
known at training time; PRF-LinAttn has no such ceiling because the
random feature map is sequence-length agnostic.

\paragraph{Kernel methods.}
Linear Transformer~\cite{katharopoulos2020transformers} replaces softmax
with $\phi(\cdot)=\mathrm{elu}+1$, recovering a recurrence at
the cost of expressivity. Performer~\cite{choromanski2021performer}
restores expressivity through positive random features but does not
explore the heavy-tailed sampling we find crucial for long-context
modelling. Our work can be read as a careful re-evaluation of
Performer at $32$K-token contexts with modern training recipes.

\paragraph{State-space models.}
S4~\cite{gu2022s4} parameterises the attention-as-convolution view via
HiPPO operators, achieving strong long-range arena results without
explicit attention. State-space approaches and linear attention are
complementary: our prefix-sum block fits inside any block that exposes
a $(Q,K,V)$ interface and can be combined with Hyena-style global
convolutions in future work.

\paragraph{IO-aware kernels.}
FlashAttention~\cite{dao2022flashattention} keeps the asymptotic
quadratic cost but eliminates the activation-memory bottleneck through
tile-wise softmax. We compare against FlashAttention-2 throughout as
the practical strong baseline at moderate context lengths.

\section{Method}
\label{sec:method}

\subsection{Preliminaries}
Let $Q, K, V \in \mathbb{R}^{n \times d}$ be the query, key, and value
projections of $n$ input tokens with head dimension $d$. Standard
softmax attention computes
\begin{equation}
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{Q K^\top}{\sqrt{d}}\Bigr) V,
\label{eq:softmax}
\end{equation}
which requires materialising the $n \times n$ score matrix at total
cost $O(n^2 d)$.

\subsection{Positive Random Feature Map}
We approximate $\exp(q^\top k / \sqrt{d})$ as the inner product
$\phi(q)^\top \phi(k)$ of two non-negative feature vectors
$\phi:\mathbb{R}^d \to \mathbb{R}^m_{\geq 0}$ given by
\begin{equation}
\phi(x) = \frac{\exp(-\|x\|_2^2 / 2)}{\sqrt{m}}
         \Bigl[\exp(w_1^\top x), \ldots, \exp(w_m^\top x)\Bigr],
\label{eq:phi}
\end{equation}
where $\{w_i\}_{i=1}^m$ are sampled iid from a centred multivariate
$t$-distribution with $\nu$ degrees of freedom. The heavier tails of
the $t$-distribution dominate the under-estimation regime that hurts
long-context softmax estimates; Section~\ref{sec:ablation} measures the
effect of $\nu$.

\subsection{Streaming Prefix Sum for Causal Masking}
The causal mask forbids token $i$ from attending to position $j>i$. We
maintain a running prefix sum $S_i = \sum_{j\leq i} \phi(k_j) v_j^\top$
and normaliser $Z_i = \sum_{j\leq i} \phi(k_j)$. Each token's output is
\begin{equation}
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top Z_i + \epsilon},
\label{eq:prefix}
\end{equation}
yielding an exact recurrence at $O(m d)$ work per token and $O(n m d)$
total. The recurrence fits in shared memory for $m\leq 512$ and $d\leq
128$, motivating the fused kernel of Section~\ref{sec:speed}.

\subsection{Complexity Analysis}
\label{sec:complexity}
The peak activation memory of softmax attention scales as $O(n^2)$ even
under FlashAttention~\cite{dao2022flashattention} (constant-factor
reduction). PRF-LinAttn replaces this with $O(n m + m d)$ activations,
matching $O(n^2)$ once $m \approx n$ but strictly better once $n \gg m$
which is the regime of interest. Wallclock cost is dominated by the
prefix-sum update rather than the projection $\phi$ because $m \ll n$;
Section~\ref{sec:speed} validates this prediction.

\section{Experiments}
\label{sec:exp}

\subsection{Setup}
\label{sec:setup}
We pretrain a $350$M-parameter decoder-only transformer with $24$
layers, hidden size $1024$, and $16$ heads (so $d=64$). The
optimiser is AdamW with peak learning rate $3{\times}10^{-4}$, cosine
decay, and a $2$K-step warm-up; weight decay is set to $0.1$. The
training corpus is a $30$B-token slice of RedPajama deduplicated at
the document level. We use RoPE~\cite{su2024roformer} as the default
positional encoding and the GPT-NeoX tokenizer. All baselines share
the optimiser, schedule, and tokenizer to isolate the attention
contribution.

\subsection{Main Results}
\label{sec:exp_main}
Table~\ref{tab:main} reports test-set perplexity on three long-context
corpora. PRF-LinAttn matches the dense FlashAttention baseline on
PG19~\cite{rae2020pg19} and outperforms it on ProofPile and the
ZeroSCROLLS~\cite{shaham2023zeroscrolls} averaged subtasks while
running $4.1{\times}$ faster.

\begin{table}[h]
\centering
\caption{Test perplexity (lower is better) on three long-context corpora
at $8$K context. All models are $350$M parameters, $30$B training
tokens. PRF-LinAttn-$m{=}256$ matches dense softmax on PG19 and beats
every linear baseline on all three corpora.}
\label{tab:main}
\begin{tabular}{lccc}
\toprule
Method                                & PG19 & ProofPile & ZeroSCROLLS \\
\midrule
FlashAttention-2~\cite{dao2022flashattention} (dense) & 11.92 & 3.81 & 18.07 \\
Performer~\cite{choromanski2021performer}             & 13.24 & 4.22 & 19.95 \\
Linformer~\cite{wang2020linformer}                    & 13.78 & 4.41 & 20.62 \\
Longformer~\cite{beltagy2020longformer}               & 12.65 & 3.97 & 18.83 \\
S4~\cite{gu2022s4}                                    & 12.41 & 3.92 & 18.55 \\
\textbf{PRF-LinAttn-$m{=}256$ (Ours)}                 & \textbf{11.89} & \textbf{3.62} & \textbf{17.41} \\
\bottomrule
\end{tabular}
\end{table}

Figure~\ref{fig:scaling} traces perplexity as a function of context
length from $1$K to $32$K tokens. The dense softmax baseline degrades
gracefully but its activation memory crashes the $48$GB device at
$32$K. Performer's perplexity diverges past $8$K, consistent with the
under-estimation hypothesis of Section~\ref{sec:method}. PRF-LinAttn
remains within $0.1$ perplexity of the dense baseline throughout the
range and is the only method that successfully evaluates at $32$K
without external tiling.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{fig_scaling.pdf}
\caption{Test perplexity versus evaluation context length on the
ZeroSCROLLS validation split. Dense FlashAttention-2 is the strongest
baseline below $16$K but exhausts $48$GB device memory at $32$K
(dashed segment). Performer diverges past $8$K. PRF-LinAttn tracks the
dense baseline within $0.1$ perplexity throughout and is the only
method that evaluates at $32$K without external tiling. Three random
seeds per method; shaded band is $\pm 1$ standard deviation.}
\label{fig:scaling}
\end{figure}

\subsection{Wallclock Speedup}
\label{sec:speed}
Figure~\ref{fig:speed} measures end-to-end forward+backward wallclock
on a single H100 at batch size $4$, head dimension $64$.
PRF-LinAttn is strictly faster than FlashAttention-2 past $4$K tokens
and the gap widens monotonically; at $32$K the speedup is $7.8{\times}$.
The slope of the linear fit on log-log axes matches the predicted
exponents: $\approx 2.0$ for FlashAttention and $\approx 1.0$ for
PRF-LinAttn, confirming the asymptotic analysis of
Section~\ref{sec:complexity}.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{fig_speed.pdf}
\caption{Forward+backward wallclock (ms) vs context length on a single
H100 at batch size $4$. Log--log axes; the slope of the linear fit
recovers the asymptotic exponent of each method (text in the upper
left). PRF-LinAttn crosses FlashAttention-2 at $\approx 4$K tokens and
reaches a $7.8{\times}$ speedup at $32$K. Three runs per point; bars
are within marker size.}
\label{fig:speed}
\end{figure}

\subsection{Ablations}
\label{sec:ablation}
Table~\ref{tab:ablation} ablates the feature dimension $m$, the
sampling distribution, and the prefix-sum precision. The dominant
factor is the feature dimension: below $m{=}128$ the linear
approximation loses $\sim 1$ perplexity point on PG19, while
$m{\geq}256$ saturates within noise of the dense baseline.
Switching the sampling distribution from Gaussian to heavy-tailed
Student-$t$ ($\nu{=}3$) reduces under-estimation in the tail and
contributes $\approx 0.2$ perplexity on ZeroSCROLLS. Single-precision
prefix sums introduce no measurable degradation up to $32$K tokens,
allowing a $2{\times}$ kernel speedup at fixed memory.

\begin{table}[h]
\centering
\caption{Ablation on PG19 ($\Delta$ from PRF-LinAttn-$m{=}256$, lower
is better). Feature dimension dominates; heavy-tailed sampling adds a
secondary improvement; single-precision prefix sums are free.}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
Configuration & PG19 PPL & $\Delta$ \\
\midrule
PRF-LinAttn-$m{=}256$ (Ours)        & 11.89 & ---    \\
\quad $m{=}64$                      & 12.92 & $+1.03$ \\
\quad $m{=}128$                     & 12.31 & $+0.42$ \\
\quad $m{=}512$                     & 11.88 & $-0.01$ \\
\quad Gaussian sampling             & 12.04 & $+0.15$ \\
\quad fp32 prefix sum               & 11.89 & $+0.00$ \\
\quad fp16 prefix sum               & 11.91 & $+0.02$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{fig_ablation.pdf}
\caption{Ablation on the feature dimension $m$. Each point averages
three random seeds at $8$K context on PG19. The dashed horizontal line
is the dense FlashAttention-2 baseline; PRF-LinAttn meets the baseline
at $m{=}256$ and slightly improves past it.}
\label{fig:ablation}
\end{figure}

\subsection{Sensitivity to Positional Encoding}
\label{sec:pos}
Repeating the main table at $8$K with ALiBi~\cite{press2022alibi}
instead of RoPE shifts every method by less than $0.1$ perplexity,
showing that the PRF-LinAttn gain is orthogonal to positional encoding.

\section{Limitations}
\label{sec:limitations}
The streaming prefix-sum is sequential along the time axis, so multi-GPU
context parallelism still requires the standard ring-attention pass.
We also do not study extreme context regimes beyond $32$K because the
RedPajama slice we use has limited documents past that length. Finally,
PRF-LinAttn inherits Performer's reliance on positive feature maps; we
expect a Fourier-feature variant could relax this, which we leave to
future work.

\section{Conclusion}
\label{sec:conclusion}
Linear attention with positive random features and a streaming
prefix sum recovers full softmax quality on long-context language
modelling while remaining $O(nd^2)$ in time and memory. The recipe is
deployable as a drop-in attention block and requires no auxiliary
losses or learned positional buckets. We release the fused CUDA kernel,
training checkpoints, and evaluation harness upon acceptance.

\bibliography{refs}
\end{document}
"""

REFS_BIB = """\
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
@article{tay2022efficient,
  title={Efficient Transformers: A Survey},
  author={Tay, Yi and Dehghani, Mostafa and Bahri, Dara and Metzler, Donald},
  journal={ACM Computing Surveys},
  volume={55},
  number={6},
  pages={1--28},
  year={2022}
}
@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNN}s: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\\c{c}}ois},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
@inproceedings{choromanski2021performer,
  title={Rethinking attention with {P}erformers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and Dohan, David and Song, Xingyou and Gane, Andreea and Sarlos, Tamas and Hawkins, Peter and Davis, Jared and Mohiuddin, Afroz and Kaiser, Lukasz and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
@article{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and Li, Belinda Z and Khabsa, Madian and Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}
@inproceedings{kitaev2020reformer,
  title={Reformer: The efficient transformer},
  author={Kitaev, Nikita and Kaiser, {\\L}ukasz and Levskaya, Anselm},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
@inproceedings{zaheer2020bigbird,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17283--17297},
  year={2020}
}
@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}
@inproceedings{gu2022s4,
  title={Efficiently modeling long sequences with structured state spaces},
  author={Gu, Albert and Goel, Karan and R{\\'e}, Christopher},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
@inproceedings{dao2022flashattention,
  title={Flashattention: Fast and memory-efficient exact attention with {IO}-awareness},
  author={Dao, Tri and Fu, Dan and Ermon, Stefano and Rudra, Atri and R{\\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={16344--16359},
  year={2022}
}
@inproceedings{press2022alibi,
  title={Train short, test long: Attention with linear biases enables input length extrapolation},
  author={Press, Ofir and Smith, Noah and Lewis, Mike},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
@article{su2024roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng and Bo, Wen and Liu, Yunfeng},
  journal={Neurocomputing},
  volume={568},
  pages={127063},
  year={2024}
}
@inproceedings{rae2020pg19,
  title={Compressive transformers for long-range sequence modelling},
  author={Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and Lillicrap, Timothy P},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
@inproceedings{shaham2023zeroscrolls,
  title={{ZeroSCROLLS}: A zero-shot benchmark for long text understanding},
  author={Shaham, Uri and Ivgi, Maor and Efrat, Avia and Berant, Jonathan and Levy, Omer},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP},
  pages={7977--7989},
  year={2023}
}
@article{liu2024lostmiddle,
  title={Lost in the middle: How language models use long contexts},
  author={Liu, Nelson F and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal={Transactions of the Association for Computational Linguistics},
  volume={12},
  pages={157--173},
  year={2024}
}
"""


THIRD_PARTY = {
    "iclr2026":    REPO / "third_party" / "iclr2026" / "iclr2026",
    "cvpr2024":    REPO / "third_party" / "cvpr2024",
    "icml2024":    REPO / "third_party" / "icml2024",
    "neurips2024": REPO / "third_party" / "neurips2024",
    "acl_arr":     REPO / "third_party" / "acl_arr",
    "arxiv_plain": None,
}


def _generate_figures(venue_dir: Path) -> None:
    """Emit fig_scaling.pdf / fig_speed.pdf / fig_ablation.pdf in ``venue_dir``.

    The numbers match the perplexity / wallclock / ablation tables in
    ``PAPER_BODY`` so the figures are internally consistent.  Real curves
    on real axes — the point is that empty ``1e-17`` plots are the
    fingerprint of a failed pipeline run, not a usable demo PDF.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ── Fig 1: PPL vs context length ──────────────────────────────────
    ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768])
    flash_ppl = np.array([12.05, 11.98, 11.94, 11.92, 11.95, np.nan])  # OOM @32K
    perf_ppl  = np.array([12.78, 12.94, 13.24, 13.71, 14.62, 16.10])
    ours_ppl  = np.array([12.01, 11.94, 11.90, 11.89, 11.91, 11.97])
    ours_band = np.array([0.06, 0.05, 0.05, 0.04, 0.05, 0.07])

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ctx, flash_ppl, "o-", color="#1f77b4", label="FlashAttention-2 (dense)", linewidth=1.8)
    # Dashed continuation past OOM for visual cue
    ax.plot(ctx[-2:], [flash_ppl[-2], flash_ppl[-2] + 0.3], "--", color="#1f77b4", alpha=0.5)
    ax.plot(ctx, perf_ppl, "s-", color="#d62728", label="Performer", linewidth=1.8)
    ax.plot(ctx, ours_ppl, "D-", color="#2ca02c", label="PRF-LinAttn (ours)", linewidth=2.0)
    ax.fill_between(ctx, ours_ppl - ours_band, ours_ppl + ours_band,
                    color="#2ca02c", alpha=0.20)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx)
    ax.set_xticklabels(["1K", "2K", "4K", "8K", "16K", "32K"])
    ax.set_xlabel("Evaluation context length (tokens)")
    ax.set_ylabel("Test perplexity (ZeroSCROLLS-val)")
    ax.set_title("Long-context perplexity vs. sequence length")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(11.5, 16.5)
    fig.tight_layout()
    fig.savefig(venue_dir / "fig_scaling.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Wallclock ms vs context, log-log w/ slope annotations ──
    flash_ms = np.array([3.2, 6.8, 14.5, 31.0, 68.0, np.nan])  # OOM
    ours_ms  = np.array([2.7, 4.1, 6.3, 9.8, 16.5, 28.4])

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.loglog(ctx, flash_ms, "o-", color="#1f77b4", label="FlashAttention-2", linewidth=1.8)
    ax.loglog(ctx, ours_ms, "D-", color="#2ca02c", label="PRF-LinAttn (ours)", linewidth=2.0)
    # Slope fits (in log-log)
    valid = ~np.isnan(flash_ms)
    slope_f = np.polyfit(np.log(ctx[valid]), np.log(flash_ms[valid]), 1)[0]
    slope_o = np.polyfit(np.log(ctx), np.log(ours_ms), 1)[0]
    ax.text(0.05, 0.92, f"FlashAttn slope ≈ {slope_f:.2f}",
            transform=ax.transAxes, color="#1f77b4", fontsize=10)
    ax.text(0.05, 0.83, f"PRF-LinAttn slope ≈ {slope_o:.2f}",
            transform=ax.transAxes, color="#2ca02c", fontsize=10)
    # Cross-over marker
    ax.axvline(4096, color="grey", linestyle=":", alpha=0.5)
    ax.text(4400, 3.2, "cross-over\n≈4K", fontsize=8, color="grey")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Forward+backward wallclock (ms)")
    ax.set_title("End-to-end speed on a single H100, batch=4")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(venue_dir / "fig_speed.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Ablation on feature dimension m ────────────────────────
    m_vals = np.array([32, 64, 128, 256, 512, 1024])
    ppl_at_m = np.array([13.85, 12.92, 12.31, 11.89, 11.88, 11.87])
    err = np.array([0.09, 0.07, 0.06, 0.05, 0.05, 0.06])

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.errorbar(m_vals, ppl_at_m, yerr=err, fmt="D-", color="#2ca02c",
                linewidth=2.0, capsize=3, label="PRF-LinAttn (ours)")
    ax.axhline(11.92, color="#1f77b4", linestyle="--", linewidth=1.6,
               label="FlashAttention-2 baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(m_vals)
    ax.set_xticklabels([str(x) for x in m_vals])
    ax.set_xlabel("Random feature dimension $m$")
    ax.set_ylabel("PG19 test perplexity")
    ax.set_title("Feature-dimension ablation (8K context, 3 seeds)")
    # Saturation annotation
    ax.annotate("saturates at $m{=}256$",
                xy=(256, 11.89), xytext=(96, 12.6),
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.6),
                fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(11.6, 14.1)
    fig.tight_layout()
    fig.savefig(venue_dir / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def stage_assets(venue_id: str, dst: Path) -> list[str]:
    src = THIRD_PARTY[venue_id]
    copied: list[str] = []
    if not src:
        return copied
    for ext in ("*.sty", "*.bst", "*.tex", "*.cls"):
        for f in src.rglob(ext):
            if "__MACOSX" in str(f):
                continue
            target = dst / f.name
            shutil.copy(f, target)
            copied.append(f.name)
    return copied


def compile_with_tectonic(workdir: Path) -> dict:
    proc = subprocess.run(
        ["tectonic", "--keep-logs", "paper.tex"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=180,
    )
    pdf = workdir / "paper.pdf"
    return {
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-4:],
        "pdf_exists": pdf.exists(),
        "pdf_bytes": pdf.stat().st_size if pdf.exists() else 0,
    }


def pdf_pages(pdf: Path) -> int | None:
    if not pdf.exists():
        return None
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        for ln in out.splitlines():
            if ln.startswith("Pages:"):
                return int(ln.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def column_check(pdf: Path) -> str:
    """Heuristic: in two-column PDFs, ``Multi-`` hyphenates within the first 12 lines."""
    if not pdf.exists():
        return "n/a"
    try:
        out = subprocess.check_output(
            ["pdftotext", "-layout", str(pdf), "-"], text=True
        )
    except Exception:
        return "n/a"
    first = "\n".join(out.splitlines()[:12])
    if "Multi-\nparagraph" in first or "Multi-" in first.split("\n")[-3:]:
        return "two-column-ish"
    # Two-column body lines are typically shorter (≈55 chars vs ≈90).
    body_lines = [
        ln for ln in out.splitlines() if 20 < len(ln.strip()) < 200
    ][:30]
    if not body_lines:
        return "?"
    avg = sum(len(ln) for ln in body_lines) / len(body_lines)
    return f"avg_line={avg:.0f}ch ({'two_col' if avg < 70 else 'single_col'})"


def main() -> int:
    out_root = Path("/tmp/full_paper_demo")
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir()

    print("=" * 78)
    print("Venue router routes 4 fixtures → expected venue")
    print("=" * 78)
    fixtures = [
        ("CV",     {"title": "Diffusion-based image detection", "claim_type": "empirical",
                    "domain": "vision", "has_real_data": True, "tier": 1,
                    "page_count_estimate": 8}),
        ("NLP",    {"title": "Cross-lingual transfer for NER", "claim_type": "empirical",
                    "domain": "nlp", "has_real_data": True, "tier": 2,
                    "page_count_estimate": 9}),
        ("ML",     {"title": "Long-context linear attention", "claim_type": "empirical",
                    "domain": "ml", "has_real_data": True, "tier": 1,
                    "page_count_estimate": 9}),
        ("Theory", {"title": "Proof of convergence",  "claim_type": "theory",
                    "domain": "theory", "has_real_data": False, "tier": 2,
                    "page_count_estimate": 14}),
    ]
    cfg = venue_router.load_venue_config()
    for tag, state in fixtures:
        result = venue_router.evaluate_venues(state, cfg)
        if result["selected"]:
            picked = result["selected"]["venue"].template_id
            score = result["selected"]["breakdown"]["score"]
            print(f"  {tag:7s} → {picked:14s}  score={score:.2f}")

    summary: list[dict] = []
    print()
    print("=" * 78)
    print("For each venue: normalize source → stage assets → tectonic compile")
    print("=" * 78)
    # Render plan: every venue gets the submission-mode build (default), and
    # the four venues that expose a submission/camera-ready toggle each get a
    # second ``camera-ready`` build (``submission_mode=False``) so reviewers
    # can eyeball the line-numbers-vs-final difference side-by-side.
    DUAL_MODE_VENUES = ("iclr2026", "neurips2024", "acl_arr", "cvpr2024")
    build_plan: list[tuple[str, str, dict]] = []
    for venue_id in sorted(list_adapters()):
        build_plan.append((venue_id, venue_id, {}))
        if venue_id in DUAL_MODE_VENUES:
            build_plan.append((venue_id, f"{venue_id}_camera_ready",
                               {"submission_mode": False}))

    for venue_id, build_id, normalize_kwargs in build_plan:
        ad = get_adapter(venue_id)
        venue_dir = out_root / build_id
        venue_dir.mkdir()
        # write refs.bib first so adapter sees real file
        (venue_dir / "refs.bib").write_text(REFS_BIB)
        stage_assets(venue_id, venue_dir)
        # ICLR also wants its preamble math_commands.tex preserved
        # ICLR + all 3 stub adapters (NeurIPS/ACL/CVPR) accept submission_mode;
        # arxiv_plain ignores unknown kwargs via its signature so this is safe.
        try:
            tex = ad.normalize_source(PAPER_BODY, **normalize_kwargs)
        except TypeError:
            tex = ad.normalize_source(PAPER_BODY)
        (venue_dir / "paper.tex").write_text(tex)
        # ICLR adapter writes the bundle via copy_files too; mirror that side.
        try:
            ad.copy_files(venue_dir)
        except Exception:
            pass
        # Generate real-data figures (replaces TAROT-style empty-axis plots).
        _generate_figures(venue_dir)
        compile_res = compile_with_tectonic(venue_dir)
        lint = lint_manuscript(source=tex, adapter=ad, page_count=8)
        summary.append({
            "venue": build_id,
            "column_layout": ad.column_layout,
            "bibstyle": ad.bibstyle_name,
            "max_pages": ad.max_pages,
            "compile_returncode": compile_res["returncode"],
            "pdf_bytes": compile_res["pdf_bytes"],
            "pdf_pages": pdf_pages(venue_dir / "paper.pdf"),
            "rendered_columns": column_check(venue_dir / "paper.pdf"),
            "lint_pass": lint["pass"],
            "lint_summary": lint["summary"],
        })

    print()
    print(f"{'venue':14s} {'col':14s} {'bib':24s} {'pages':>5s} {'pdfB':>7s} {'render':28s} lint")
    print("-" * 110)
    for row in summary:
        print(
            f"{row['venue']:14s} {row['column_layout']:14s} "
            f"{row['bibstyle']:24s} "
            f"{str(row['pdf_pages']):>5s} "
            f"{row['pdf_bytes']:>7d} "
            f"{row['rendered_columns']:28s} "
            f"{'PASS' if row['lint_pass'] else 'FAIL'}"
        )

    bundle_path = out_root / "demo_summary.json"
    bundle_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"[ok] summary written to {bundle_path}")
    print(f"[ok] PDFs under {out_root}/<venue>/paper.pdf")
    return 0 if all(r["compile_returncode"] == 0 for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
