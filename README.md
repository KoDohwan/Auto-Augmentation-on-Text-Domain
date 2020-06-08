## Executing Stanford Corenlp Server

```
cd stanford-corenlp-full-2018-10-05
java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000
```

[Download Stanford Corenlp](https://stanfordnlp.github.io/CoreNLP/)

You must execute **Stanford Corenlp Server** berfore augmentation.

**Control + C to quit executing server.**  



## Glove

Download [glove.840B.300d.zip](https://nlp.stanford.edu/projects/glove/) and unzip it to .txt file.

Put that file in glove folder.   



## Configuration

Open **config.py**, and change the line which was annotated.   



## Augmentation

``` 
python augment.py
```



## Training Model

``` 
python model.py   
```





