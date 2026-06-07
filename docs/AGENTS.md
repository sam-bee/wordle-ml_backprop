# Instructions for Agents on the Wordle Game Engine

This repo is a game engine for Wordle, written in Go. It is part of the Wordle ML project.

## Go Guidelines

The targetted language version is 1.26. You may not alter this.

## Systems Administration

You are absolutely forbidden from doing your own systems administration. If an important tool is not installed on
the system, or you can't access something you need, STOP. Ask the user to install tools or change your access
privileges. You MAY NOT install additional tools on the development mahcine. You MAY NOT seek creative solutions
to systems access problems. You MUST stop and ask if you are missing something important.

## Docs

Any documentation necessary should be in the `docs/` folder, in markdown by default.

## Version Control

The two remotes are called `gitlab` and `github`. They should typically be kept in sync, and you should be pushing to
both.

When writing a commit message, prefer this format:

`To/Because/For/So that [reason for change], [nature of change]`.

However, be sensible: a larger changeset may require more than a single-sentence explanation.

## Historical Background

This code was originally a Wordle player, which played Wordle well, based on dictionary files. It has been copied here,
complete with its version control history. It is now being changed to be a separate, reusable game engine. This is
because it is needed for a more recent Go project - specifically a machine learning project. Ultimately, the code and
README.md will reflect its new purpose. Do not get confused if this is still in progress.
