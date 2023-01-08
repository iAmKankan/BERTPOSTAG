## Parts of Speech Tagging using BERT
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

## PROBLEM STAEMENT
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)
In this project we will be performing one of the most famous task in the field of nautal language processing i,e Parts of Speech Tagging using BERT.

## DESCRIPTION OVERVIEW
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)
Part-Of-Speech tagging (POS tagging) is also called grammatical tagging or word-category disambiguation. It is the corpus linguistics of corpus Text data processing techniques for marking meaning and context.

Part-of-speech tagging can be done manually or by a specific algorithm. Using machine learning methods to implement part-of-speech tagging is the research content of Natural Language Processing (NLP). Common part-of-speech tagging algorithms include Hidden Markov Model (HMM), Conditional Random Fields (CRFs), and so on.

Part-of-speech tagging is mainly used in the field of text mining and NLP. It is a preprocessing step for various types of text-based machine learning tasks, such as semantic analysis and coreference resolution.


1. CC Coordinating conjunction
2. CD  Cardinal number
3. DT  Determiner
4. EX Existential there
5. FW Foreign word
6. IN Preposition or subordinating conjunction
7. JJ  Adjective
8. JJR  Adjective, comparative
9. JJS  Adjective, superlative
10. LS  List item marker
11. MD  Modal
12. NN  Noun, singular or mass
13. NNS  Noun, plural
14. NNP  Proper noun, singular
15. NNPS  Proper noun, plural
16. PDT  Predeterminer
17. POS  Possessive ending
18. PRP  Personal pronoun
19. PRP$  Possessive pronoun
20. RB  Adverb
21. RBR  Adverb, comparative
22. RBS  Adverb, superlative
23. RP  Particle
24. SYM  Symbol
25. TO  to
26. UH  Interjection
27. VB Verb, base form
28. VBD  Verb, past tense
29. VBG  Verb, gerund or present participle
30. VBN  Verb, past participle
31. VBP  Verb, non-3rd person singular present
32. VBZ  Verb, 3rd person singular present
33. WDT  Wh-determiner
34. WP  Wh-pronoun
35. WP$  Possessive wh-pronoun
36. WRB  Wh-adverb

## TECHNOLOGY USE
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

Here we will be using  Anaconda Python 3.6 , Pytorch 1.4 with GPU support CUDA 10 with CuDNN 10.

## INSTALLATION
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

Installation of this project is pretty easy. Please do follow the following steps to create a virtual environment and then install the necessary packages in the following environment.

**In Pycharm it’s easy** 

1. Create a new project.
2. Navigate to the directory of the project
3. Select the option to create a new new virtual environment using conda with **python3.6**
4. Finally create the project using used resources.
5. After the project has been created, install the necessary packages from **requirements.txt** file using the command _`pip install -r requirements.txt`_


**In Conda also it’s easy**

1. Create a new virtual environment using the command
    _`conda create -n your_env_name python=3.6`_
2. Navigate to the project directory.
3. Install the necessary packages from **requirements.txt** file using the command         
_`pip install -r requirements.txt`_

## WORKFLOW DIAGRAM
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

## IMPLEMENTATION
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187187-8a435135-3c0f-4e85-bd5e-895280eafe56.png" width=60%/>
</p>


### 1. Project Directory
![light](https://user-images.githubusercontent.com/12748752/181097751-9be22081-c630-4756-9ea8-2c27fdce6984.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187233-91c36ab0-ccd3-4afb-b98c-d0b4007488cb.png" width=60%/>
</p>


This above picture shows the folder structure of the project. Here project folder consists of data and BERT models. 

### 2. bertlayr.py
![light](https://user-images.githubusercontent.com/12748752/181097751-9be22081-c630-4756-9ea8-2c27fdce6984.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187314-8efb957f-5f69-40b3-9a38-1e247ac1e967" width=60%/>
</p>

This file consists of the the bert model architecture which will be used to train the data.

### 3. sentPosTagger.py

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187336-79e6581b-7029-4d94-bbdf-82b01c750ba4.png" width=60%/>
</p>

This file is used to train the model and to do the prediction.

### 4. trainCustomPostagger.py
![light](https://user-images.githubusercontent.com/12748752/181097751-9be22081-c630-4756-9ea8-2c27fdce6984.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187366-05c9451b-d6c6-4422-bde7-c476a673c971.png" width=60%/>
</p>


This file is used to train a custom pos tagging model if the user wants to train.

### 5. downLoadlibs.py
![light](https://user-images.githubusercontent.com/12748752/181097751-9be22081-c630-4756-9ea8-2c27fdce6984.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187387-fd0d4b6f-9e7c-4623-9da2-9027024d8960.png" width=60%/>
</p>

This file is used  to download dataset.

### 6. ClientApp.py
![light](https://user-images.githubusercontent.com/12748752/181097751-9be22081-c630-4756-9ea8-2c27fdce6984.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187409-a56e9640-23df-42f4-8bbf-910e87e5d1e3.png" width=60%/>
</p>


This is tha flask server file.

## TESTING IN LOCAL/API
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

To do the test testing we need to run the clientApp.py and after that web server will start at **http://0.0.0.0:5000/**

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187461-f5572ad1-8d59-4be5-8f8b-a31d4866d3da.png" width=60%/>
</p>

Enter the sentence and click on predict button.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187534-a9a6057b-9ad6-4045-a22b-88e3300bb9df.png" width=60%/>
</p>

After clicking predict

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187500-9289be96-5ebd-4e07-96ae-11e5ba4b7229.png" width=60%/>
</p>

Results are shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/211187488-d5ccd748-699d-47d2-a8e1-4977872c4f66.png" width=60%/>
</p>

Do the matching of the words with corresponding colours.

## CONCLUSION
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

Here we successfully performed Parts of Speech tagging on the given dataset.
## COMPARISION
![deep](https://user-images.githubusercontent.com/12748752/181097747-f97a41d2-ebab-4295-8dae-fac47563a251.png)

More data or better larger dataset can be used to build a better model. We can also try out better pre trained model with fine tuning to increase the performance.
