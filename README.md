# CV-Neural-Style-Transfer

### Demo

![image](https://user-images.githubusercontent.com/34294344/81485980-64fd4380-9206-11ea-80e4-aea3c4c797ff.png)


### Gallery

| Content & Style  | Generated |
| ------------- | ------------- |
| <img width="300" src="https://user-images.githubusercontent.com/34294344/81485250-8ce9a880-9200-11ea-87e6-11893e082453.png">  | <img src="https://user-images.githubusercontent.com/34294344/81485284-c1f5fb00-9200-11ea-8af3-8c81f4372181.jpeg"> |
| <img width="300" src="https://user-images.githubusercontent.com/34294344/81485264-9ffc7880-9200-11ea-8dc1-198a1942e28b.png">  | <img src="https://user-images.githubusercontent.com/34294344/81485299-e18d2380-9200-11ea-8bdd-b814bcd9ce6d.jpeg" width="400">  |
<img width="300" src="https://user-images.githubusercontent.com/34294344/81485271-af7bc180-9200-11ea-95d8-d16ce1b08223.png"> | <img src="https://user-images.githubusercontent.com/34294344/81485302-f2d63000-9200-11ea-84da-f0a3e4b4ee23.jpeg">


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
