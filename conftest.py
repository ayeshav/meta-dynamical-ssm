"""Repo root conftest: makes the in-tree `meta_ssm` package importable in tests.

The project has no installed package / pyproject, so pytest's default
import mode does not put the repo root on sys.path. Pytest adds the
directory containing the top-most conftest.py to sys.path, so this file
existing here is enough for `import meta_ssm` to resolve.
"""
