# SANER2021-DocumentationSmell-ReplicationPackage
## Automatic Detection of Five API Documentation Smells: Practitioners’ Perspectives

## Documentation Smell
Documentation smells can be described as bad documentation styles that do not necessarily make a documentation incorrect but make it difficult to understand and use.


## Types
We present 5 types of documentation smells. They are:
* Bloated: too lengthy and verbose.
* Excess Structural Info: too many structural syntax or information
* Tangled: too complex to read and understand
* Fragmented: scattered over multiple pages or sections
* Lazy: does not provide extra info other than the method prototype


## Survey
We conducted a survey of 21 software developers to validate these documentation smells. The survey questionnaire and responses can be found in the 'Survey' folder. 


## Benchmark Dataset
We created a benchmark dataset of 1000 documentations with these 5 types of smells. The benchmark dataset with the features can be found in the 'Benchmark Dataset' folder.


## Automatic Detection of Documentation Smells
We employed rule-based, shallow, and deep machine learning techniques to automatically detect documentation smells.

## Codes
Codes of our rule-based, shallow, and deep learning classifiers are available at three different subfolders under the 'Codes' folder. The codes are organized in a sequential manner. All file&folder names are self-explanatory. We have run the deep learning classifiers (i.e., Bi-LSTM, BERT) on Google Colab. You can run the corresponding python notebooks on Google Colab by uploading them on your google drive with the benchmark dataset.
