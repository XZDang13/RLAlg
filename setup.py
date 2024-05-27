from setuptools import setup, find_packages

# Read the long description from the README file
#with open("README.md", "r", encoding="utf-8") as fh:
#    long_description = fh.read()

setup(
    name="RLAlg",
    version="0.1.0",
    author="XDang",
    author_email="xdang13@outlook.com",
    description="RL Algorithm in PyTorch",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/XZDang13/RLAlg.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        # List your package dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Example: 'your-script = RLAlg.module:function'
        ],
    },
)
