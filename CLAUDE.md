# marmalade-tts-cli — project notes for Claude

## Cross-repo conventions live in the core repo

The rules that bind **every** marmalade repo — secrets, wire contract, security
invariants, testing — are a linked documentation tree in the core `marmalade`
repo (checked out at `~/coding/marmalade/marmalade`). **Enter at
`docs/README.md` and follow the links down.** Those pages are authority: don't
reinvent a decision that already has one.

The rule that bites this repo: **credentials resolve through a configurable
command whose documented default is `marmalade secret get <entry>`** — never a
bespoke key file, env var, or hardcoded password-manager call. The Venice
engine's `api_key_file` (`~/.config/marmalade-tts/venice-api-key`, 0600)
predates that rule and is tracked for migration to a
`venice/api-key` keyring entry; see `docs/conventions/secrets.md` in the core
repo. **Do not add a third mechanism in the meantime.**

## Remotes

**github is the authoritative remote.** Push only to `github` (`git push github main`).

```
github   https://github.com/maxwhipw/marmalade-tts.git    (authoritative)
origin   http://george:3000/marmalade/marmalade-tts-cli.git  (Forgejo mirror)
```

**Do NOT `git push origin` by hand.** Forgejo is intended to auto-mirror from github
via Forgejo's "Pull mirror" feature (repo settings → mirror settings). Until Max
sets that up, Forgejo will fall behind and must be brought into sync manually only
when needed — never by routine push, which risks recreating the parallel-history
divergence that happened in May 2026.

If you need to bring Forgejo in line with github before the Pull mirror is set up,
the safe sequence is:
1. Archive Forgejo's current main as a branch on github (`git push github
   <archive-branch>`) so nothing is lost.
2. `git push origin +main` to force-align (the `+` refspec syntax is the
   non-`--force` form the harness permits).

The first time this happened, the divergence was wide (zero common ancestor, 17
unique commits on Forgejo all functionally superseded by github's chain) — and the
archive is at `archive/forgejo-history` on github if anyone needs to look at the
old story.
