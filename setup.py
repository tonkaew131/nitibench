from setuptools import setup, find_packages

setup(
    name="lrg",
    version="0.1.0",
    author="Pawitsapak Akarajaradwong",
    author_email="pawitsapaka_visai@vistec.ac.th",
    description="LRG research project.",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'notebook-clear = tools.notebook_clear:main'
        ]
    }
) 