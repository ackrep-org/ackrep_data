# ACKREP Constitution

## Preamble

This document describes how individuals interact in the scope of the project "Automatic Control Knowledge Repository" (ACKREP). This project only makes sense if people are willing to contribute their work. Therefore a focus on equal participation and maintaining a constructive and welcoming atmosphere is central. The main purpose of this constitution is to ensure that the project is driven by equal peers and not harmfully dominated by an individual or a small group.

## Decision making (scalable rejection-minimizing consensus mechanism)

All decisions concerning the project are taken by the whole group of recognized contributors in form of a poll. All contributors have the right to suggest a poll, to add poll options during the discussion phase and to vote on a poll (during the voting phase). The poll allows to express approval (+1, +2), neutrality (0), or rejection (-1, -2, -3) for each poll option. Approval and rejection are recorded separately (i.e. they **do not compensate**). The lowest value (-3) is considered as **veto**. The winning option is the one with the weakest rejection, among those without veto. In case of ambiguity stronger approval rate is relevant. If ambiguity persists, the poll is repeated with just the ambiguous options. If there is no option without veto, then every veto-caster has the possibility to explain their concerns (again). Everyone has the right to withdraw or add options to the poll (second discussion phase). Then the voting is repeated (second voting phase). If there is still no option without veto, then up to a fraction of 10.0% (w.r.t the total numbers of recognized contributors) vetos are admissible.

Discussion phase(s) and voting phase(s) are announced via email at least 48 hours in advance.
The votes of recognized contributors which do not participate in a poll, are counted as neutral.

### Transparency of decision making

For later reference every formal decision will be recorded as a protocol inside the repository.

## Contribution Process

An official contribution to the Automatic Control Knowledge Repository is a commit to the `main`-branch of its underlying git repo. This repo is currently (during development) hosted at <https://github.com/cknoll/ackrep_data>.

The regular way for a prospective contributor (P) to get a commit merged into the `main`-branch is to file a *merge-request* via the the ACKREP web service. This triggers the service to clone the contributors version of the repository, run all automated tests and report the results to P and all previous official contributors. Optionally some review communication along with updates to the newly contributed branch takes place. Finally, the group of previous contributors holds a poll whether to accept the merge request. In case of acceptance one of the maintainers then merges the contribution. In any case the protocol of the request along with the poll decision is commited.

It is the contributors own responsibility to add the own name and email adress to the list of contributors and making this change part of the contribution.

## Conflict Prevention
The contributors recognize that disagreement among humans is more the default rather than a special case, and that disagreement comes with the risk of conflict, also for intelligent and educated individuals. Therefore, all contributors confirm that in every ACKREP-related communication or action they regard the individual respect of other people involved. Criticism has to be formulated on a factual level.


## Declaratory Effect of this Constitution

To simplify the development of this project, this constitution obtains declaratory effect if the list of recognized contributors contains five or more individuals. Earlier decisions can be made informally but should be oriented as much as possible on this document.


---


# Remarks

## General
The current state of the constitution is very basic. However, it aims to be sufficient to enable its own evolution in whatever direction the group of contributors deems it right to evolve.

## Decision making
The defined scheme makes it possible to find among many (probably similar) options the one which raises the least resistance inside the group. This facilitates future cooperation. The acceptable fraction of vetos should ensure scalability as the group grows, while the second discussion phase should ensure that no important concern remains unheard. If the group of contributors should grow

