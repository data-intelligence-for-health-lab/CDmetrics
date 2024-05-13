from setuptools import setup, find_packages

setup(
    name="case_difficulty_metrics",
    version="0.0.1",
    description="Case difficulty calculation metrics package written by DIH",
    author="EuleeKwon",
    author_email="euleekwon@gmail.com",
    url="https://github.com/data-intelligence-for-health-lab/d_metrics",
    install_requires=["openxl", "pandas", "scikit-learn", "tensorflow", "keras"],
    packages=find_packages(exclude=[]),
    keywords=["data_difficulty", "case_difficulty", "instance_hardness", "difficulty"],
    python_requires=">=3.6",
    package_data=find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
