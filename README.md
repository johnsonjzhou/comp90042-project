# Automated Fact Checking For Climate Science Claims

Johnson Zhou (Student ID: 1302442)

Project under COMP90042 Natural Language Processing at the University of Melbourne. Semester 1, 2023. Please refer to the project [requirements][proj_req].

## Dependencies

This project was conducted on an Apple MacBook Pro with M2-max processor and 96 GB of RAM. The included `environment.yml` provides the basic operating environment for **Apple Silicon** devices. The default python version is **3.8**. For `pytorch` models, usage of the Metal Performance Shader (`mps`) device is preferred and selected where available.

```shell
# Initialise environment (comp90042_project)
conda env create -f ./environment.yml
```

### Additional dependencies: spaCy

The following dependencies were installed for `spacy` functionality. The `en_core_web_trf` pipeline is used due to higher accuracy.

```shell
pip install -U pip setuptools wheel
python -m spacy download en_core_web_trf
```


[proj_req]: ./doc/project.pdf