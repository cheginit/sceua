---
name: update-package
description: Update this package's locked dependencies and pre-commit hooks, then land the change. `pixi update` refreshes pixi.lock, `pixi r pcupdate` moves the hook revisions, the full gate from AGENTS.md runs with the dependency audit, findings the newer tools raise get fixed at their cause, and the change goes in through a PR merged once CI passes. Use when asked to update dependencies, refresh or bump the lock file, update the pre-commit hooks, or run /update-package.
---

# Update dependencies and hooks

Asking for this skill is the go-ahead to push the branch, open the PR, and merge it once
CI passes, for a contributor allowed to merge; anyone else stops at the open PR and asks
a maintainer. It covers nothing else: no tag, no release, no force-push, no change to
repository settings, and no merge while a required check is red or pending.

`$SCRATCH` below is a scratch directory outside the repository.

## 1. Preflight

Stop at the first failure and report it. The tree must be clean and match `origin/main`,
since the PR has to hold the update and nothing else:

```bash
git fetch -q origin
DEF=$(git symbolic-ref --short refs/remotes/origin/HEAD | sed 's#^origin/##')
test "$(git branch --show-current)" = "$DEF" &&
  test -z "$(git status --porcelain)" &&
  test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$DEF")"
```

Then branch, following the `<type>/<short-description>` rule in AGENTS.md:
`B=chore/update-deps-$(date +%F); git switch -c "$B"`, adding `-2` for a second run the
same day.

## 2. Update

```bash
pixi update --json >| "$SCRATCH/pixi-update.json"
pixi r pcupdate >| "$SCRATCH/pcupdate.log" 2>&1
python3 .agents/skills/update-package/summarize.py "$SCRATCH/pixi-update.json"
```

The script prints one Markdown row per change (package, before, after, environments) and
the change count on stderr; a report with no changes prints `0 package changes`. A row
with the same version on both sides is a build-string change. The hook log names each
hook that moved (`updating v0.16.10 -> v0.17.1`). Mark the changes to the runtime
dependencies, `[project] dependencies` in `pyproject.toml` (`numpy` and `scipy`), since
reviewers read those first. When neither `pixi.lock` nor `.pre-commit-config.yaml`
changed, report that, delete the branch, and stop.

## 3. Gate

Delete `.coverage` and any `.coverage.*` files first, since coverage accumulates across
runs (AGENTS.md, Gotchas). Then run, each to a clean exit:

```bash
pixi r lint && pixi r typecheck && pixi r -e test313 test && pixi r -e test315 test &&
  pixi r docs && pixi r audit
```

- Lint runs the new hook versions over every file. When a formatter rewrites files, run
    lint once more and keep the rewrite for its own commit.
- The docs build and the audit belong here even with no change under `docs/`: a new
    zensical or mkdocstrings can break the build, and a new resolution can pull in an
    advisory.

## 4. Fix what the new versions raise

Fix each finding at its cause. Never silence one to pass: no new ignore, no lowered
strictness in `pyproject.toml`, no skipped or loosened test, and no package pinned back
without the maintainer's yes. A finding that needs a design call, or a package whose new
version breaks something with no small fix, goes to the maintainer with the failing
output and the options: fix it, pin an upper bound, or hold the package out by updating
the others by name (`pixi update <pkg> ...`). A fix that makes or reverses a design
decision updates `docs/design.md` in the same change.

## 5. Commit

One commit per concern, staged by path, each chained to the gate with nothing between
the gate and `&&`:

1. `chore: update pixi.lock`, the body giving the change count and the runtime changes;
1. `chore: update pre-commit hooks`, the body naming each hook's new revision;
1. a formatter rewrite from the new hooks: `style: apply <tool> <version> formatting`;
1. each fix, as `fix:` or `refactor:`, naming what the new version changed.

Check each with `git show --stat HEAD` before making the next.

## 6. Pull request and merge

Push the branch and open the PR from `.github/pull_request_template.md`: the runtime
changes, the hook revisions, the full table in a collapsed `<details>` block, and the
gate results, the audit included. With no issue to close, say so on the body's first
line in place of `Closes #`.

Wait for CI without a polling loop where your agent's environment forbids one; in a
plain terminal, `gh pr checks <n> --watch --fail-fast` blocks until the checks finish.
The ruleset on `main` requires `Test (test313)`, `Test (test315)`, `typecheck`, `lint`,
and `build`. A red check is fixed on the branch and pushed, back through steps 3 to 5.

Merge only when every required check passes and the PR reads CLEAN, with a rebase merge:
`gh pr merge <n> --rebase`. Record the branch tip first (`TIP=$(git rev-parse "$B")`),
then:

```bash
git fetch -q origin
test "$(git rev-parse "origin/$DEF^{tree}")" = "$(git rev-parse "$TIP^{tree}")" &&
  git switch -q "$DEF" && git merge -q --ff-only "origin/$DEF" && git branch -D "$B"
```

A tree mismatch means `main` moved during the merge: report it and keep the local
branch. Report the PR, the merged commit, and the notable changes in a few lines.

## Pre-flight before the merge

Answer in writing: the branch holds only the update and its fixes; every gate step
passed on the final tree, the docs build and the audit included; no finding was
silenced; the runtime changes are named in the PR; every required check passed and the
PR is CLEAN.
