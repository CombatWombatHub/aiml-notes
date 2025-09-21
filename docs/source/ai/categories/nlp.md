# Natural Language Processing
[Natural Language Processing (NLP)](https://www.geeksforgeeks.org/nlp/natural-language-processing-nlp-tutorial/) understands and interacts with human languages in a way that feels natural. `NLP` uses [ML](./ml.md), but isn't [ML](./ml.md) itself. `NLP` is the application, [ML](./ml.md) is the engine.
- [Kaggle Natural Language Processing Guide](https://www.kaggle.com/learn-guide/natural-language-processing) has links to a lot of relevant tutorials and project ideas. The best place to start is probably their guide on [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)

## Maintenance Records Classification Task
- Trying to remember and practice a particular problem
- dataset was perhaps four columns, I sort of remember 3 of them
- need to predict the category from the other data
- first order of business: find a similar dataset or problem
- first place to start is probably the Kaggle Guide [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners).

| date | service | category |
| --- | --- | --- |
| 4/13/2004 | patched hull | watercraft |
| 6/1/2005 | tire change | vehicle |
| 6/22/2005 | waxed wing, swapped landing gear spring | aircraft |
| ... | ... | ... |

## Transformers
- the Kaggle [Getting started with NLP for absolute beginners](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) notebook required some additional libraries to be installed
- one of them was `transformers` from [Hugging Face](https://huggingface.co/docs/transformers/en/index)
- it's a "model-definition framework" 
- sounds like a standardized way of using models so that you could, say, take a model that was trained in `Tensorflow` and load it into `PyTorch`? Or have either of those things reach out to a trained `LLM`?