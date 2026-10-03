<p align="center">
  <img src="docs/assets/deepgraph-hero.svg" alt="DeepGraph: an autonomous research agent that improves by being judged by reality" width="100%">
</p>

<p align="center">
  <b>The autonomous research agent of JouleBeat, an open-source RSI lab.</b><br>
  It reads the literature, finds open questions, runs the experiments,<br>
  and lets reality decide what is true.
</p>

<p align="center">
  <a href="https://deepgraph.joulebeat.com">Website</a> ·
  <a href="docs/RESULTS.md">Results</a> ·
  <a href="docs/SHOWCASE.md">Showcase</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/loop-Harness%20%C3%97%20Benchmark-F5B841" alt="Harness × Benchmark">
</p>

---

## Why DeepGraph

**JouleBeat builds self-improving AI by letting harnesses and benchmarks evolve together.**

How an agent works (its *harness*: tools, memory, workflow, verification) now moves capability by as much as a model generation, and open source has made harness variants cheap to generate. What is still scarce is the *benchmark* that tells real progress from a lucky score. So we put both in the loop: harnesses compete under benchmarks, and benchmarks compete on how well they predict results that arrive later.

That loop needs a steady stream of real tasks whose answers can be checked. Computational research is exactly that: a conclusion is confirmed or refuted by rerunnable code, held-out data or an independent recomputation, in minutes to days. **DeepGraph is where those tasks come from.**

- **For researchers**, it is a research agent that delivers conclusions you can check.
- **For us**, it is the accelerator of our own R&D: every task is a judged trial for the next harness and the next benchmark.

## How it works

| | Step | What happens |
|:-:|---|---|
| 1 | **Read** | Harvests arXiv at scale and extracts claims, methods and results |
| 2 | **Map** | Merges them into an evidence graph of entities, relations and contradictions |
| 3 | **Ask** | Ten structural detectors find open questions in pure SQL, with no LLM in the path |
| 4 | **Test** | Turns a question into a pre-registered, budgeted experiment on CPU or GPU |
| 5 | **Judge** | An evidence ladder issues a verdict only when the comparison is fair, and the result updates what gets searched next |

Every verdict is auditable, and the system audits itself: in August 2026 it found and retracted 11 of its own conclusions that rested on empty model outputs. A search loop is only as good as the filter that decides what is true. Details in [Results](docs/RESULTS.md).

## At a glance

Live production data, August 2026.

| | |
|---|---|
| Papers | 24,407 harvested, 7,005 through the full evidence pipeline |
| Evidence graph | 248,441 entities, 735,913 relations, 238 contradiction clusters |
| Discovery | 25,652 mapped research opportunities |
| Experiments | 150 audited verdicts across six research agendas |
| Compute | 921M LLM tokens invested |

## The JouleBeat stack

| Layer | What it is | Status |
|---|---|---|
| **DeepGraph** | Research agent: real tasks in, checkable conclusions out | In service, with paying research groups |
| **Harness × Benchmark libraries** | Versioned harnesses, and benchmarks scored on how well they predicted later outcomes | Harness versioning running; benchmark validation in build |
| **[Evolution Kernel](https://github.com/Protocol-zero-0/evolution-kernel)** | The engine: propose, evaluate, accept or roll back | Open source, `pip install evolution-kernel` |

## Quick start

Python 3.12+ and an LLM API key are the minimum; PostgreSQL enables the full control plane.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set at least DEEPGRAPH_LLM_API_KEY
export $(grep -v '^#' .env | xargs)
python3.12 main.py      # open http://localhost:8080
```

Full deployment (PostgreSQL, systemd, remote GPU backends): [docs/DEPLOY.md](docs/DEPLOY.md).

## Learn more

- [Architecture](docs/ARCHITECTURE.md): design principles, the loop, evidence gates, repository map, tests
- [Results](docs/RESULTS.md): every measured outcome, read directly from the production database
- [Showcase](docs/SHOWCASE.md): demo route and case studies
- [Roadmap](docs/ROADMAP.md)

## Contributing

Researchers: bring a computational question at [deepgraph.joulebeat.com](https://deepgraph.joulebeat.com). Developers: issues and pull requests are welcome; contributions are covered by the [CLA](CLA.md).

## License

[MIT](LICENSE)
