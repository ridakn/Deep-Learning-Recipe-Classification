# Deep Learning Recipe Classification
Developed a predictive model that can determine, given a recipe, which of the 12 categories the recipe falls in.

## Introduction 

<p align = "justify">
Text classification is a popular problem in the deep learning field with various possible applications and abundant implementations already in place. Classification in text is a common task of classifying a sequence into a single category. This could be sentiment of a sentence, marking an email as spam or inappropriate comments getting flagged among other examples. Classification problems could be binary or multiclass. Deep neural networks, particularly recurrent networks, are being used for the purpose of text classification also known as sequence learning. In this project, the main objective was to learn to train a recurrent neural network from scratch for the purpose of classifying recipes into one of 12 categories. The input to the model would be text sequences and the output would be the class of the given recipe between 1 and 12. </p>

## Model Structure

<p align = "justify">
I started the experiments with a simple model with three LSTM layers. The initial model was overfitting which could be seen through the validation and training loss. I added a dropout layer to reduce overfitting and improve the performance. To further improve the performance, I added a bidirectional layer to the LSTM layers. The forward and backward outputs are concatenated together. Since our dataset wasn't that big, I decided to try GRU layers instead of LSTM since they're known to work better and faster for smaller datasets. Replacing the LSTM layers with GRU layer worked well. I then added a second bidirectional GRU layer which still further improved performance; however, adding anymore layers did not help. Finally, I tried using branch method wherein I had two identical branches: a left and right branch. Each branch consists of an embedding layer followed by a dropout layer followed by two bidirectional GRU layers and finally a dense layer as a last layer to each branch. The first bidirectional GRU layer was set to return sequences. The final dense layers of both branches had ReLU activation function. The outputs of the branches are concatenated and sent to the final layer of the model which used a SoftMax activation function and outputs the final output for the model. </p>

<img width="724" alt="Screen Shot 2021-06-22 at 2 08 08 PM" src="https://user-images.githubusercontent.com/32781544/122906539-1d9d1e00-d307-11eb-977b-a67064d53830.png">

## Classification Sample

Following is a sample prediction for the first recipe in the test set. 
The predicted category is printed below the recipe.

<img width="1010" alt="Screen Shot 2021-05-18 at 1 39 29 PM" src="https://user-images.githubusercontent.com/32781544/122907444-014db100-d308-11eb-8af2-ff106119286d.png">

