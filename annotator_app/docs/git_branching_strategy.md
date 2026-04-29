# Git Branching Strategy for Application Development

## Purpose

This document defines the Git branching strategy for an application with three major development needs:

1. A stable public version of the app
2. A short-term stripped-down testing version
3. Experimental feature development

The goal is to keep the public app stable while allowing temporary testing modifications and experimental work without creating long-term branch confusion.

---

## High-Level Strategy

Use the following branch structure:

```text
main
develop
stripped-testing
feature/*
experiment/*
bugfix/*
hotfix/*
```

Recommended workflow summary:

```text
Normal feature:
feature/* → develop → main

Experiment:
experiment/* → feature/* → develop → main

Temporary stripped-down testing app:
main → stripped-testing

Shared bug fix:
bugfix/* → main → stripped-testing

Urgent production fix:
hotfix/* → main → develop
```

---

## Branch Definitions

## 1. `main`

### Purpose

`main` is the stable public/current version of the application.

This branch represents the version of the app that is safe for public use and production deployment.

### Rules

- `main` should always be deployable.
- No unfinished or experimental work should be committed directly to `main`.
- Changes should be merged through pull requests.
- Tests and linting should pass before merge.
- This branch should be protected if using GitHub/GitLab/Bitbucket branch protection.

### Deployments

```text
Public app deployment source: main
```

---

## 2. `develop`

### Purpose

`develop` is the integration branch for upcoming stable work.

Completed feature branches should be merged into `develop` first. After testing and review, `develop` can be merged into `main`.

### Rules

- `develop` can be slightly less stable than `main`, but should still be functional.
- Do not merge rough experiments directly into `develop`.
- Only merge features that are intended to eventually reach production.
- Use this branch to test the next stable version of the app.

### Workflow

```text
feature/<feature-name> → develop → main
```

Example:

```text
feature/video-upload → develop → main
```

---

## 3. `stripped-testing`

### Purpose

`stripped-testing` is a temporary branch for a simplified version of the app that removes or disables key features for a specific testing purpose.

This branch is being used instead of feature flags because:

- The stripped-down version is short-term.
- The changes are not intended to remain in the main codebase.
- We want to avoid cluttering the main app with temporary testing-only code.
- We do not want long-term configuration complexity for a one-off testing version.

### Base Branch

Default base branch:

```text
main → stripped-testing
```

If the stripped-down testing version requires unreleased but stable upcoming features, it may instead be based on:

```text
develop → stripped-testing
```

However, the preferred default is to branch from `main`.

### Rules

- Keep this branch short-lived.
- Do not use this branch for unrelated feature development.
- Do not merge stripped-down changes back into `main` unless they are intentionally reusable.
- Keep a clear list of removed, hidden, or disabled features.
- Delete the branch after the testing period is complete.
- If a bug affects both the public app and stripped-down app, fix it in `main` first, then merge or cherry-pick the fix into `stripped-testing`.

### Recommended Workflow

Create the branch:

```bash
git checkout main
git pull origin main
git checkout -b stripped-testing
git push -u origin stripped-testing
```

Apply stripped-down modifications on `stripped-testing`.

For shared bug fixes:

```text
bugfix/<bug-name> → main → stripped-testing
```

Example:

```bash
git checkout main
git checkout -b bugfix/fix-login-error

# make changes
git add .
git commit -m "Fix login error"
git push -u origin bugfix/fix-login-error

# open PR: bugfix/fix-login-error → main
```

After the bug fix is merged into `main`:

```bash
git checkout stripped-testing
git pull origin stripped-testing
git merge main
git push origin stripped-testing
```

Alternatively, cherry-pick only the needed commit:

```bash
git checkout stripped-testing
git cherry-pick <commit-hash>
git push origin stripped-testing
```

### Deployment

```text
Stripped-down testing app deployment source: stripped-testing
```

---

## 4. `feature/*`

### Purpose

`feature/*` branches are used for normal production-intended features.

Each feature branch should represent one focused unit of work.

### Naming Convention

```text
feature/<short-description>
```

Examples:

```text
feature/video-upload
feature/export-results
feature/user-auth
feature/annotation-review
feature/admin-dashboard
```

### Rules

- Branch from `develop`.
- Keep the branch focused on one feature.
- Merge back into `develop` through a pull request.
- Delete the branch after it is merged.
- Do not use feature branches for long-term variants of the app.

### Workflow

Create a feature branch:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/video-upload
git push -u origin feature/video-upload
```

Merge path:

```text
feature/video-upload → develop → main
```

---

## 5. `experiment/*`

### Purpose

`experiment/*` branches are used for exploratory or uncertain work.

These branches are for testing ideas that may or may not become part of the final application.

### Naming Convention

```text
experiment/<short-description>
```

Examples:

```text
experiment/realtime-feedback
experiment/new-ui-layout
experiment/llm-generated-comments
experiment/scoring-v2
experiment/alternate-review-workflow
```

### Rules

- Branch from `develop` unless the experiment specifically needs to be based on `main`.
- Experiments should not merge directly into `main`.
- Experiments should not merge directly into `develop` unless they have been cleaned up and reviewed.
- If an experiment becomes useful, convert it into a proper `feature/*` branch.
- Delete abandoned experiments when they are no longer needed.

### Workflow if Successful

```text
experiment/<idea> → feature/<feature-name> → develop → main
```

Example:

```text
experiment/realtime-feedback → feature/realtime-feedback → develop → main
```

### Workflow if Unsuccessful

```text
experiment/<idea> → abandoned/deleted
```

---

## 6. `bugfix/*`

### Purpose

`bugfix/*` branches are used for non-emergency bug fixes.

### Naming Convention

```text
bugfix/<short-description>
```

Examples:

```text
bugfix/fix-login-error
bugfix/fix-video-upload-validation
bugfix/fix-form-submission
```

### Rules

- Branch from the branch where the bug exists.
- If the bug affects production, branch from `main`.
- If the bug affects upcoming development only, branch from `develop`.
- Merge bug fixes into the appropriate target branch through a pull request.

### Workflow for Production Bug

```text
bugfix/<bug-name> → main → develop
```

If the bug also affects the stripped-down testing version:

```text
bugfix/<bug-name> → main → stripped-testing
```

---

## 7. `hotfix/*`

### Purpose

`hotfix/*` branches are for urgent production fixes that need to go into `main` quickly.

### Naming Convention

```text
hotfix/<short-description>
```

Examples:

```text
hotfix/fix-crash-on-launch
hotfix/fix-broken-deployment
hotfix/fix-critical-auth-error
```

### Rules

- Branch from `main`.
- Merge back into `main` as soon as the fix is reviewed.
- After merging into `main`, also merge `main` back into `develop`.
- If relevant, merge or cherry-pick the fix into `stripped-testing`.

### Workflow

```text
hotfix/<issue> → main → develop
```

Optional:

```text
main → stripped-testing
```

---

## Recommended Branch Layout

```text
main
develop
stripped-testing
feature/video-upload
feature/export-results
feature/user-auth
experiment/realtime-feedback
experiment/new-scoring-model
bugfix/fix-login-error
hotfix/fix-critical-auth-error
```

---

## Deployment Strategy

## Public App

```text
Deploy from: main
Purpose: stable current app
```

## Stripped-Down Testing App

```text
Deploy from: stripped-testing
Purpose: temporary simplified testing version
```

## Experimental App

```text
Deploy from: experiment/<experiment-name>
Purpose: prototype or test new ideas
```

## Development/Staging App

```text
Deploy from: develop
Purpose: test upcoming stable features before release
```

---

## Merge Rules

## Into `main`

Allowed sources:

```text
develop
hotfix/*
bugfix/*
```

Rules:

- Require pull request review.
- Require tests to pass.
- Require linting to pass.
- No direct commits.
- No experimental branches should merge directly into `main`.

---

## Into `develop`

Allowed sources:

```text
feature/*
bugfix/*
main
```

Rules:

- Feature branches should be reviewed before merging.
- Experiments should only enter `develop` after being cleaned up into a `feature/*` branch.
- Keep `develop` functional.

---

## Into `stripped-testing`

Allowed sources:

```text
main
bugfix/*
hotfix/*
```

Rules:

- Use `main` as the source of shared fixes.
- Avoid merging `stripped-testing` back into `main`.
- Keep stripped-testing-specific changes isolated.
- Delete the branch after the testing purpose is complete.

---

## Into `experiment/*`

Allowed sources:

```text
develop
main
feature/*
```

Rules:

- Experiments can pull in updates from `develop` as needed.
- Avoid letting experiments become long-running forks.
- If useful, convert the experiment into a feature branch.

---

## Coding Agent Instructions

When implementing this branching strategy, follow these instructions:

1. Create or verify the existence of the following core branches:

```text
main
develop
stripped-testing
```

2. Ensure `main` is treated as the stable production branch.

3. Ensure `develop` is used as the integration branch for production-intended features.

4. Create `stripped-testing` from `main` unless there is a specific need to include unreleased work from `develop`.

5. Do not introduce feature-flag or configuration infrastructure solely for the stripped-down testing version unless it is already useful for the long-term app.

6. Keep stripped-down testing changes isolated to the `stripped-testing` branch.

7. Do not merge `stripped-testing` back into `main`.

8. For new production-intended features, create branches using:

```text
feature/<feature-name>
```

9. For exploratory work, create branches using:

```text
experiment/<experiment-name>
```

10. If an experiment becomes production-intended, clean it up in a `feature/*` branch before merging into `develop`.

11. For bugs affecting production, create:

```text
bugfix/<bug-name>
```

from `main`.

12. For urgent production fixes, create:

```text
hotfix/<issue-name>
```

from `main`.

13. After fixing production bugs or hotfixes, propagate the fixes to `develop` and, if needed, to `stripped-testing`.

14. Delete short-lived branches after they are merged or no longer needed.

15. Keep branch names lowercase and hyphen-separated.

Good branch names:

```text
feature/video-upload
feature/export-results
experiment/realtime-feedback
bugfix/fix-login-error
hotfix/fix-deployment-crash
```

Avoid vague branch names:

```text
feature/stuff
experiment/test
fixes
new-version
```

---

## Suggested Initial Setup Commands

Starting from an existing repository:

```bash
git checkout main
git pull origin main

# Create develop if it does not already exist
git checkout -b develop
git push -u origin develop

# Create stripped-testing from main
git checkout main
git pull origin main
git checkout -b stripped-testing
git push -u origin stripped-testing
```

If `develop` already exists:

```bash
git checkout develop
git pull origin develop

git checkout main
git pull origin main
git checkout -b stripped-testing
git push -u origin stripped-testing
```

---

## Pull Request Expectations

Each pull request should include:

- Summary of changes
- Reason for the change
- Branch being merged into
- Testing performed
- Screenshots or demo notes, if UI-related
- Any known limitations

Suggested PR template:

```markdown
## Summary

Briefly describe what this PR changes.

## Target Branch

Merging into:

- [ ] main
- [ ] develop
- [ ] stripped-testing
- [ ] other: ___

## Type of Change

- [ ] Feature
- [ ] Bug fix
- [ ] Hotfix
- [ ] Experiment
- [ ] Stripped-down testing change
- [ ] Refactor
- [ ] Documentation

## Testing

Describe what was tested.

## Notes

Include any risks, limitations, or follow-up work.
```

---

## Branch Protection Recommendations

For `main`:

- Require pull requests before merging.
- Require at least one approval.
- Require passing tests.
- Require passing linting.
- Prevent force pushes.
- Prevent direct commits.

For `develop`:

- Require pull requests before merging.
- Require tests if available.
- Prevent force pushes.

For `stripped-testing`:

- Branch protection is optional because it is temporary.
- If multiple people are working on it, require pull requests.
- Prevent accidental merges back into `main`.

---

## End-of-Testing Cleanup for `stripped-testing`

After the stripped-down testing period is complete:

1. Confirm no important reusable code exists only on `stripped-testing`.
2. If reusable changes exist, extract them into a proper `feature/*` or `bugfix/*` branch.
3. Do not merge the entire `stripped-testing` branch back into `main`.
4. Delete the remote branch:

```bash
git push origin --delete stripped-testing
```

5. Delete the local branch:

```bash
git branch -d stripped-testing
```

If the branch has unmerged temporary changes and Git refuses to delete it:

```bash
git branch -D stripped-testing
```

---

## Final Recommended Strategy

Use this branch model:

```text
main
develop
stripped-testing
feature/*
experiment/*
bugfix/*
hotfix/*
```

Core principles:

- `main` is stable and public.
- `develop` is for upcoming stable features.
- `stripped-testing` is a short-term branch for the simplified testing app.
- `feature/*` branches are for focused production-intended work.
- `experiment/*` branches are for exploratory ideas.
- `bugfix/*` and `hotfix/*` branches are for maintenance.
- Shared fixes should flow from `main` into `develop` and `stripped-testing` as needed.
- `stripped-testing` should not become a permanent fork.
- Delete temporary branches once they are no longer needed.
