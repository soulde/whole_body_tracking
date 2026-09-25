# Contributing

This repository follows a Git Flow-style workflow. Commit messages, pull
request titles, and issue titles are written in English; documentation, pull
request bodies, and code comments are written in English.

## Branches

- `main` contains released, stable versions.
- `develop` is the integration branch and must remain buildable.
- Use short-lived `feat/<kebab-case>`, `fix/<kebab-case>`, or
  `chore/<kebab-case>` branches created from `develop`.
- Reserve `release/vX.Y.Z` and `hotfix/<kebab-case>` for release work.

## Commits and pull requests

Use Conventional Commits with an imperative, lowercase subject of no more
than 50 characters:

```text
<type>(<scope>): <subject>
```

Valid types are `feat`, `fix`, `refactor`, `docs`, `test`, `build`, `ci`, and
`chore`. Pull requests target `develop` for normal work and `main` only for
releases or hotfixes. Include background, changes, and verification sections;
require at least one non-author approval and passing checks before squash
merging.

## Verification

Run the smallest relevant checks locally. Isaac Lab and Isaac Sim commands
must run inside the pinned project Docker image, with the image digest,
repository commit, configuration, and seed recorded in the experiment
manifest.
