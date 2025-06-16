"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: benchmark/git_checks.py
Description: File contains utility functions that check if we have uncommited changes and funciton to get curreent 
             branch info.

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""


from git import Repo
import os


def check_git_status() -> tuple[bool, bool]:
    """
    Check the git status and return True, if the working tree is clean

    returns: [has_staged_changes, has_unstaged_changes]
    """
    repo_parent = os.path.dirname(__file__)
    repo = Repo(repo_parent, search_parent_directories=True)

    # 1. Check for staged changes (ready to commit)
    staged = repo.index.diff("HEAD")  # staged vs HEAD
    has_staged_changes = bool(staged)

    # 2. Check for unstaged changes (working tree vs index)
    unstaged = repo.index.diff(None)  # index vs working tree
    has_unstaged_changes = bool(unstaged)

    return has_staged_changes, has_unstaged_changes


def get_git_info() -> tuple[str, str, str]:
    """
    Get branch and commit hash

    :returns: <branch name>, <commit hash>, <commit message>
    """
    repo_parent = os.path.dirname(__file__)
    repo = Repo(repo_parent, search_parent_directories=True)

    branch_name = repo.active_branch.name
    commit_hash = repo.active_branch.commit.hexsha
    commit_msg = repo.active_branch.commit.message

    return branch_name, commit_hash, commit_msg
