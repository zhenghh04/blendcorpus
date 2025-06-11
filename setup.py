from setuptools import setup, find_packages

setup(
    name='llm_dataset',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'preprocess_data=preprocess.preprocess_data:main',
        ]
    },
    scripts=['preprocess/tokenization.sh'],
    install_requires=[
        'nltk',
        'torch',
        'numpy',
        'megatron'
    ],
)
