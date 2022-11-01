# dl4cv -  A Computer Vision Hugging Face Server Backend :rocket:

This is a projekt to integrate the major Computer Vision Use Cases into one Backend. The Deep Learning is  done via the
Hugging Face Models.

This project was generated via [manage-fastapi](https://ycd.github.io/manage-fastapi/)! :tada:


  - [Showcases](#showcases)
  - [Installation](#installation-of-dependencies)
  - [Usage](#use-the-api)


## Showcases

### Image Classifikation


### Image Segmentation


### Object Detection



### Panoptic Segmentation


### KeyPointDetection


## Installation of Dependencies
Conda/Miniconda is recommended. This Project uses poetry for dependency management. To install poetry run:

```pip install poetry```

To install the dependencies run:

```poetry install```

To activate your poetry Venv run:

```poetry shell```



## Use the API
To start the API in dev Mode use:
```uvicorn main:app --reload```

## API Docs
```localhost:8000/docs```
