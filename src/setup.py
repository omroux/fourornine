from setuptools import setup, find_packages

setup(
    name="fourornine",
    version="0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
    ],
    entry_points={
        "console_scripts": ["fourornine=fourornine.__main__:main"]
    }
)
