from setuptools import setup, find_packages

setup(
    name="ml_optimization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "matplotlib",
        "jupyter",
        # Add other dependencies from requirements.txt
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine Learning Optimization Project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-optimization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 