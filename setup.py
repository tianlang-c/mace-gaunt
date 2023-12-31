from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


setup(
    name='MACE',
    version='0.0.1',
    description='',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.8'],
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8',
        'numpy',
        'ase',
        'torch_geometric>=1.7.1',
    ],
    zip_safe=False,
    test_suite='pytest',
    tests_require=[
        'pytest',
        'sympy',
    ],
)