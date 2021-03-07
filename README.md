# CNN - multiclass-classification

Model’s architecture:
I chose to create CNN model which consists of 4 convolution layers, such that at each layer I used the
Relu activation function and max pool for decreasing the output matrix’s size. Moreover, I added
dropout to my network in order to prevent over fitting. The last part of my whole network is a fully
connected neural network with 3 layers (2 hidden layers) such that the output layer’s size is 30 as the
number of the classes we had to classify.


Hyper parameters and model’s properties:
The Filters sizes of the convolution layers: for the first 2 convolution layers I choose the filters size to
be 4X4, whereas for the two last convolution layers I choose the filters sizes to be 5X5. The sizes were
tested by me and finally I chose the filter sizes with the best performance. The filters have stride in
size 1 and zero padding value of 2.
After examining different learning rate, I choose a learning rate of 0.001 which come up with the
Adam optimizer for my model. I used Adam optimizer because it performed the best results for my
model’s predictions.
More details: I chose a batch size of 200 (after trying batch sizes like: 50, 100, 150). The number of
epochs for the training loop of the model is 20, I examined more epochs size like: 15 and 25 but I saw
that 20 performs high results and do not tend to over fitting.
I chose to use 4 convolution layers because more than 4 convolution layers took too many time and
was not needed because the model’s predictions accuracy was high enough. Moreover, more number
of any type of layers (convolution or fully connected) may lead to over fitting of the model.
By using all the above explained hyper parameters I received around 92% accuracy of my model’s
predictions while testing my model on the supplied validation data.


How to run my code:
Firstly, I used Google Colab framework in order to run my code faster by using the GPU’s benefits of
Google Colab. I managed to upload the full data files to my google drive and then access them from
the code in colab. I used “cuda” to utilize the GPU.
After running the code at Google Colab I used Pycharm IDE. The data files path are (at Pycharm IDE):
1) Train files path: files/train (files is a directory in my pycharm project and train is the train files with
the sub folders).
2) validation files path: files/valid
3) Test files path: files/sub_dir/test
(Inside the code’s implementation the path to load the test files is files/sub_dir/, so the code search
for sub folders inside sub_dir and the only one sub folder is the test folder).

