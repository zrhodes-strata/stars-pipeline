"""
conftest.py
===========
Root pytest configuration.

Prevents pytest from collecting test_* functions from the
stars_pipeline source package (stars_pipeline/stars/tests.py
contains public API functions named test_* that are not pytest tests).
"""
collect_ignore_glob = ["stars_pipeline/**/*.py"]
