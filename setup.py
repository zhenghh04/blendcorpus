from setuptools import setup, find_packages

setup(
    name='llm_dataset',
    version='0.1',
    packages=find_packages(include=[
        'preprocess', 'preprocess.*',
        'llm_dataset', 'llm_dataset.*',
    ]),
    package_dir={'': '.'},
    entry_points={
        'console_scripts': [
            'preprocess_data=preprocess.preprocess_data:main',
            'get_meta_data=preprocess.get_meta_data:main',
        ]
    },
    scripts=['preprocess/tokenization.sh', 'preprocess/launcher.sh'],
    install_requires=[
        'nltk',
        'torch',
        'numpy',
        'deepspeed',
        'ezpz @ git+https://github.com/saforem2/ezpz.git',
        'sentencepiece',
        'mpi4py', 
        'zstandard',
        'pybind11'
    ],
)
