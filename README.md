# NeuralNews
A machine learning project that categorizes news articles from headlines using a fine-tuned DistilBERT model, trained on nearly 210,000 news articles from 2012 to 2022 from [Huffpost](https://www.huffpost.com/).

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Overview
[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) is a distilled version of [BERT](https://huggingface.co/docs/transformers/model_doc/bert) that retains 97% of BERT's language understanding whilst being 60% smaller and significantly faster. The pre-trained model has been trained on the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle and consists of around 210k headlines.