# Measured outcomes

Everything below is read directly from the production database. Counts are as of
2026-08-25. No number here is an estimate or a projection.

## Corpus

| quantity | value |
|---|---|
| papers harvested | 24,407 |
| carried through extraction, claims and evidence graph | 7,005 |
| queued for processing | 16,443 |
| permanently errored | 949 |

## Verdicts

The system has recorded 150 `OutcomeRecord`s across six research agendas.

| verdict | count | meaning |
|---|---|---|
| `refuted` | 61 | the system measured its own hypothesis and rejected it |
| `invalid` | 57 | the experiment itself failed; no scientific claim is made |
| `inconclusive` | 33 | the evidence did not license a verdict either way |
| `supported` | 1 | the hypothesis survived the evidence ladder |

Eleven of the `invalid` records were `refuted` or `invalid` until 2026-08-25,
when they were retracted; see below.

Refutation dominating is a property of the evidence ladder, not an accident. A
verdict is withheld when the comparison is unfair, when the baseline is absent
or zero, or when the measured effect is not real. `refuted` is itself a claim
and is subject to the same gates, which is why an unfair comparison yields
`inconclusive` rather than `refuted`.

`invalid` records are execution failures -- a run that never produced a
comparable measurement. They are kept rather than discarded so the harness's own
failure rate stays visible. Reducing this number is an engineering objective,
not a scientific one; the 2026-08-25 semantic-binding guard removes one of its
larger causes by refusing to fund a plan that no registered runner implements.

### Eleven verdicts have been retracted

An audit of every run whose raw predictions are still on disk (105 of 195) found
eleven outcome records, across four runs, whose verdict rested on an arm that
generated nothing at all: 200 of 200 predictions empty. Scoring an empty string
is honest arithmetic -- it scores zero -- and the conclusion is still fiction.
Run 235 walked the whole evidence ladder to `refuted` at p = 0.000999 that way.

Eight of them were refutations. On 2026-08-25 all eleven were retracted: the
verdict became `invalid`, the evidence state became `unmeasurable_retracted`,
and the reason and the path to the raw predictions were written into each
record. They were not deleted, because the count of what the system got wrong is
itself a measurement, and a history that quietly loses its errors cannot be
audited. The refutation count above fell from 69 to 61 for this reason.

That is a lower bound: 90 older runs no longer have artifacts to audit.

The gate that stops it is now in the pipeline in two places -- the evidence
audit refuses to compute a verdict from an arm above the blank-prediction rate,
and the pilot-promotion gate refuses to fund a full benchmark behind one. Both
fired on 2026-08-25, refusing three promotions with `pilot arm measured
nothing`. No new record of this kind can be produced.

## The supported result

`experiment_run 246`, object-counting literature agenda:

| field | value |
|---|---|
| metric | `numeric_accuracy` |
| baseline | 0.32 |
| candidate | 0.63 |
| effect | +0.31 |
| verdict | `supported` |
| evidence state | `manuscript_allowed` |

The candidate roughly doubled accuracy on a held-out split and cleared every
gate on the evidence ladder, including the manuscript gate.

## The principal negative result

On the scaffold self-evolution agenda, 26 completed experiments produced a best
gain of **+1.5 points** over the frozen baseline, and no run reached the
threshold for a supported verdict.

The honest reading: searching prompt-and-scaffold structure alone, against a
frozen small model on a 200-sample generative-QA suite, is close to exhausted as
a source of accuracy gains. This bounds a claim rather than refuting the
approach -- cost-side objectives (the same accuracy at fewer tokens) occupy a
different part of the same search space and have not yet been measured.

## Lifecycle validation

On 2026-08-25 a five-paper canary ran the full production lifecycle -- PDF
retrieval, extraction, claim mining, graph entities and relations, per-stage
checkpoints, 33 metered child usages totalling 439,494 tokens -- and settled
with zero orphan grants, jobs or reservations.

## Measurement floors

A generative-QA plan below 200 evaluation samples cannot support a verdict; the
floor is the minimum detectable effect measured in the 2026-08-17 probe at
power 0.8. Numeric-answer datasets are scored by comparing the final number on
both sides, because string equality against free-form generation measures
formatting rather than correctness.
