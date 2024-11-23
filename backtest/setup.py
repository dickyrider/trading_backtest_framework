from setuptools import setup, find_packages

setup(
    name='backtest',
    version='0.1',
    packages=find_packages(),
    description='A simple backtesting framework',
    author='Dicky',
    author_email='dickyrider@gmail.com',
    url='https://github.com/dickyrider/my_backtest',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

