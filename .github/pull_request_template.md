Closes #

## What changed

<!-- What the change does and why, in a few sentences. Name the D-, L-, or Q- entries in
docs/design.md it touches. -->

## How it was checked

<!-- Tick what ran. For a change to the search, add solution quality before and after on
fixed problems and seeds, with the prediction made first. -->

- [ ] `pixi r lint`
- [ ] `pixi r typecheck`
- [ ] `pixi r -e test313 test`
- [ ] `pixi r -e test315 test`
- [ ] `pixi r docs` (needed when `docs/`, `README.md`, or `zensical.toml` changed)

## Risk and revert

<!-- What this could break and for whom, the review depth REVIEW.md asks for, and how to
undo it: a revert of this PR, or, after a release, a fix release. -->

## Design record

- [ ] `docs/design.md` is updated, or this change makes, reverses, or answers nothing
    there
