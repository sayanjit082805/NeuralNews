# NeuralNews
A machine learning project that categorizes news articles from headlines using a fine-tuned DistilBERT model, trained on nearly 210,000 news articles from 2012 to 2022 from [Huffpost](https://www.huffpost.com/).

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Overview
[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) is a distilled version of [BERT](https://huggingface.co/docs/transformers/model_doc/bert) that retains 97% of BERT's language understanding whilst being 60% smaller and significantly faster. The pre-trained model has been trained on the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle and consists of around 210k headlines.

## Screenshots

![](https://raw.githubusercontent.com/sayanjit082805/NeuralNews/main/demo/ss1.png)

![](https://raw.githubusercontent.com/sayanjit082805/NeuralNews/main/demo/ss2.png)

## Dataset 
The dataset used has been taken from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

- `category`: Category in which the article was published.
- `headline`: Headline of the article.
- `authors`: Authors of the article.
- `link`: Link to the original article.
- `short_description`: Short description of the article.
- `date`: Date when the article was published.

There are a total of 42 categories.

![](https://raw.githubusercontent.com/sayanjit082805/NeuralNews/main/demo/categories.png)

[Dataset Source](https://rishabhmisra.github.io/publications)

## Metrics
The model achieved an accuracy of nearly 65%, on both the validation and test data.

## Citations 
1. Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

> [!NOTE]
> The License does not cover the dataset. It has been taken from Kaggle.

