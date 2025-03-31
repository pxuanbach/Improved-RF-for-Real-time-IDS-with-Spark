# Improved-RF-for-Real-time-IDS-with-Spark

## Prerequisites

Ensure you have the following installed before proceeding:

-   Docker
-   Docker Compose
-   Python 3.11

## Installation

**Note:** Run the following commands in Git Bash for Window environment:

```sh
make install
make infras
```
Linux comand: 
```
    pyenv versions
    pyenv local 3.11.9
    python -m venv venv
    source venv/bin/activate

    make install
    make infras

	python improved-ids/modules/linux-version/spark_data_preprocessing_relieff.py 
	python improved-ids/modules/linux-version/RF/spark_rf_training.py 
	python improved-ids/modules/linux-version/RF/spark_rf_detection.py 
```