# Agenda specifications

A research agenda is a direction, a budget and a scope. These are the specs the
running agendas were created from, kept in the repository because the direction
text is what decides which candidates the ideation layer is allowed to propose --
it is closer to an experimental protocol than to configuration.

Create one with:

    scripts/agenda_backends.py create --spec docs/agendas/<name>.json

Each spec states the claims it was derived from. `reject` is as load-bearing as
`prefer`: V1 registers two runners, `generative_qa` and
`sequence_classification`, whose candidate hook is the prompt or the input text,
so a direction that invites weight updates, architecture changes or code
execution produces candidates that are refused at preflight after the ideation
cost has already been paid.
