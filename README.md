# CV-Neural-Style-Transfer
#### Implementation of A Neural Algorithm of Artistic Style By Leon A. Gatys, Alexander S. Ecker, Matthias Bethge


### Demo

![image](https://user-images.githubusercontent.com/34294344/81486037-e94fc680-9206-11ea-8241-904b3e29bdb0.png)

### How to use Streamlit App
```
1. Choose a content image
2. Choose a style image
3. On left sidebar, click `Generate`
4. Adjust hyperparameters if needed (default learning_rate = 0.02)
```

### How to run this demo
The demo requires Python 3.6 or later (TensorFlow is not yet compatible with later versions). We suggest creating a new virtual Python 3.6+ environment, then running:
```
git clone https://github.com/DeepNeuralAI/CV-Neural-Style-Transfer.git
cd CV-Neural-Style-Transfer
pip install -r requirements.txt
streamlit run app.py
```

### Gallery

| Content & Style  | Generated |
| ------------- | ------------- |
| <img width="300" src="https://user-images.githubusercontent.com/34294344/81485250-8ce9a880-9200-11ea-87e6-11893e082453.png">  | <img src="https://user-images.githubusercontent.com/34294344/81485284-c1f5fb00-9200-11ea-8af3-8c81f4372181.jpeg"> |
| <img width="300" src="https://user-images.githubusercontent.com/34294344/81485264-9ffc7880-9200-11ea-8dc1-198a1942e28b.png">  | <img src="https://user-images.githubusercontent.com/34294344/81485299-e18d2380-9200-11ea-8bdd-b814bcd9ce6d.jpeg" width="400">  |
<img width="300" src="https://user-images.githubusercontent.com/34294344/81485271-af7bc180-9200-11ea-95d8-d16ce1b08223.png"> | <img src="https://user-images.githubusercontent.com/34294344/81485302-f2d63000-9200-11ea-84da-f0a3e4b4ee23.jpeg">


### Theory

_Derived from `deeplearning.ai Convolutional Networks`_

Given a content image $C$ and style image $S$, extract features from a convolutional neural network at various layers to generate an artistic image $G$



#### Architecture & Transfer Learning

Similar to the original paper, we will use the VGG network, specifically VGG-19.



VGG-19 has been trained on the ImageNet database and can recogize both low level images at the lower layers and high level images at the deeper layers.



#### NST Algorithm

1. Build a content cost function: $J_{content}(C,G)$

2. Build a style cost function: $J_{style}(S,G)$

3. Total cost function is a linear combination of content and style cost functions: $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$.



$\alpha$ -- importance of the content cost

$\beta$ -- importance of the style cost

Both are hyperparameters that can be tuned and can also incorporate normalization constants.



#### Content Cost Function

In this stage, we want to effectively forward propagate both $C$ and $G$ as input to the pretrained VGG network.



We know that the shallow layers of a ConvNet will tend to detect low level features such as edges, shapes, and textures. Conversely, we know that that deeper layers of a ConvNet tend to detect more complex objects, or class object themselves.



The paper describes that to get decent visual results, the middle layers provide the most visually pleasing results.



Steps:

- forward propagate $C$ and extract $a^{(C)}$ (hidden layer activation)

- forward propagate $G$ and extract $a^{(G)}$



We define the content cost function as:

$$J_{content}(C,G) = \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1} $$


The key takeaways are:

* content cost takes an activation from a hidden layer and measures how different $a^{(C)}$ and $a^{(G)}$ are

* we ultimately want to minimize this difference



#### "Style" (Gram Matrix)

In mathematical terms, we will define style as the correlation between activations across channels.



Intuitively, high level texture components will tend to occur together if they are correlated.



In order to measure _style_, we will construct a _Gram matrix_. Recall that in linear algebra, a Gram matrix is the matrix of dot projects.



${\displaystyle G_{ij} = v_{i}^T v_{j} = np.dot(v_{i}, v_{j}) }$.



We essentially use $G_{ij}$ to compare how similar $v_i$ is to $v_j$.



* Large dot product values implies high style similarity

* Small dot product values implies low style similarity



<img src="https://user-images.githubusercontent.com/34294344/81509995-9425ba80-92c3-11ea-82bf-67a72020a074.png" style="width:400px;height:300px;">



#### $G_{(gram)i,j}$: correlation

The result is a matrix of dimension $(n_C,n_C)$ where $n_C$ is the number of filters (channels). The value $G_{(gram)i,j}$ measures how similar the activations of filter $i$ are to the activations of filter $j$.



#### $G_{(gram),i,i}$: prevalence of patterns or textures

* The diagonal elements $G_{(gram)ii}$ measure how "active" a filter $i$ is.

* For example, suppose filter $i$ is detecting vertical textures in the image. Then $G_{(gram)ii}$ measures how common vertical textures are in the image as a whole.

* If $G_{(gram)ii}$ is large, this means that the image has a lot of vertical texture.




By capturing the prevalence of different types of features ($G_{(gram)ii}$), as well as how much different features occur together ($G_{(gram)ij}$), the Style matrix $G_{gram}$ measures the style of an image.



#### Style Cost Function



In order to build a style cost function, our objective is to minimize the distance beteen the Gram style matrix and Gram generated matrix.



For a single layer, we can define the style cost as:



$$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2\tag{2} $$



* $G_{gram}^{(S)}$ Gram matrix of the "style" image.

* $G_{gram}^{(G)}$ Gram matrix of the "generated" image.



#### Total Cost



We define the total cost function as:

$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$



Using gradient descent, in this case _Adam_, we can minimize $J(G)$, effectively updating pixel values rather than weights or parameters.



Using this cost function, we can generate an image $G$ that combines the _content_ of image $C$ and the _style_ of image $S$


### Credit

Modified code from [Style Transfer](https://www.tensorflow.org/tutorials/generative/style_transfer)

Inspired from [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

