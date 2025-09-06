from setuptools import setup, find_packages

setup(
    name='blendcorpus',
    version='0.1',
    packages=find_packages(include=[
        'preprocess', 'preprocess.*',
        'blendcorpus', 'blendcorpus.*',
    ]),
    package_dir={'': '.'},
    entry_points={
        'console_scripts': [
            'preprocess_data=preprocess.preprocess_data:main',
            'tokenization=preprocess.preprocess_data_parallel:main',            
            'get_meta_data=preprocess.get_meta_data:main',
            'test_dataloader=tests.test_dataloader:main',
            'merge_parquet=preprocess.fuse_files.merge_parquet:main'
        ]
    },
    scripts=['utils/launcher.sh', 'preprocess/tokenization/tokenization.sh', 'utils/barrier.sh', 'utils/download-huggingface-dataset.sh'],
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
