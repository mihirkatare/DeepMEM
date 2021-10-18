from setuptools import setup

extras_require = {}
extras_require["lint"] = sorted({"flake8", "black[jupyter]"})
extras_require["all"] = sorted(set(sum(extras_require.values(), [])))

setup(extras_require=extras_require)
