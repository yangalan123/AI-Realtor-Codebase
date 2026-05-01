from setuptools import setup, find_packages

with open("./requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name='llm_bargaining',
    version='0.1',
    package_dir={"llm_bargaining": "."},
    # packages=['llm_bargaining'] + ['llm_bargaining.' + p for p in find_packages(where=".")],
    packages=find_packages(where="."),
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=requirements,
)
