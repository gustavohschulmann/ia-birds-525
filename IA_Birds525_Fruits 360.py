# -*- coding: utf-8 -*-
"""

# **Classificação de imagens**

# **Introdução:**

A aprendizagem de máquina tem se tornado cada vez mais relevante e presente em diversas áreas, permitindo o desenvolvimento de sistemas inteligentes capazes de realizar tarefas complexas, como a classificação de imagens.

Nesse contexto, o presente trabalho tem como objetivo implementar duas redes neurais do tipo Convolutional Neural Network (CNN), para fazer a classificação de dois datasets distintos, aplicando técnicas de pré-processamento, como redimensionamento e normalização, a fim de melhorar a qualidade e a representatividade dos conjuntos de dados.

Iremos buscar identificar as melhores configurações de hiperparâmetros para cada dataset, comparando diferentes combinações de camadas, neurônios, funções de ativação, funções de erro e otimizadores. Os resultados obtidos serão analisados e discutidos no final, destacando o impacto de cada hiperparâmetro nos resultados e fornecendo conclusões sobre as melhores configurações alcançadas.

# Datasets

## Birds 525 Species

O primeiro dataset escolhido foi Birds 525 Species, na qual formado por imagens de 525 espécies diferentes de aves, capturadas em diversos ambientes e posições. Composto por 84635 imagens de treino, cujo 2625 teste e 2625 de validação.Todas as imagens são 224 X 224 X 3 imagens coloridas em formato jpg.

Com intuito de garantir um melhor treinamento para nossa rede, estaremos focando em 30 espécies que tem características diversas entre si, sendo elas:

Referência dos Imagens: https://www.kaggle.com/datasets/gpiosenka/100-bird-species

## Fruits 360

O segundo dataset escolhido foi Fruits 360, na qual formado por 131 classes de
 frutas e vegetais. Composto por  67692  imagens de treino, cujo 22688 teste.Todas as imagens são 100 X 100 X 3 imagens coloridas em formato jpg.

Com intuito de garantir um melhor treinamento para nossa rede, estaremos focando em 40 classes que tem características diversas entre si, sendo elas:

Referência dos Imagens: https://www.kaggle.com/datasets/moltean/fruits

# Implementação

Nesta seção, discutiremos o processo de implementação da Convolutional Neural Network (CNN) para a tarefa de classificação de imagens. A CNN é uma arquitetura de rede neural amplamente utilizada em problemas de visão computacional, conhecida por sua capacidade de extrair automaticamente características relevantes das imagens.

## Baixando os datasets

A primeira etapa consistiu em baixar os datasets selecionados para o trabalho,  para isso obtivemos Kaggle API credentials,seguindo as instruções abaixo:  

    * Go to the Kaggle website and sign in to your account (or create a new one).
    * Navigate to the "Account" section of your Kaggle profile.
    * Scroll down to the "API" section and click on the "Create New API Token" button.
    * This will download a JSON file named "kaggle.json" containing your API credentials.

Em seguida adicionamos o kaggle.json em /content.
</br>

Agora podemos instalar a biblioteca do Kaggle e configurar as credenciais da API do Kaggle no ambiente de desenvolvimento
"""

!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle --version

"""Em seguida podemos baixar os datasets bird 525 especies  e  fruits 360"""

!kaggle datasets download -d gpiosenka/100-bird-species --force
!kaggle datasets download -d moltean/fruits --force

"""Então faremos a extração do conteúdo dos arquivos compactados (zip), e guardaremos em seus respectivos diretórios."""

!unzip -o 100-bird-species.zip -d birdies
!unzip -o fruits.zip -d fruits

"""## Criando as labels das categorias para ambos datasets

Antes de prosseguir com o processamento dos datasets, é necessário filtrar as classes específicas com as quais desejamos trabalhar. Isso envolve criar uma lista de rótulos (labels) que representam as classes selecionamos previamente a partir do restante do conjunto de dados. Essa etapa de filtragem é importante para focar na análise e classificação das classes que escolhemos, tornando o processo mais direcionado e relevante para nosso objetivo.

 Essa abordagem nos permite concentrar nossos esforços em treinar e avaliar modelos de classificação específicos para as frutas escolhidas, tornando o processo mais eficiente e eficaz
"""

fruit_labels = [
    'Apple Crimson Snow',
    'Strawberry',
    'Tamarillo',
    'Tomato 1',
    'Walnut',
    'Watermelon',
    'Pepper Green',
    'Lemon',
    'Lychee',
    'Mandarine',
    'Mango',
    'Avocado',
    'Carambula',
    'Cauliflower',
    'Cucumber Ripe',
    'Dates',
    'Eggplant',
    'Pineapple',
    'Mulberry',
    'Kiwi',
    'Kaki',
    'Maracuja',
    'Pitahaya Red',
    'Physalis',
    'Raspberry',
    'Redcurrant',
    'Tangelo',
    'Fig',
    'Onion White',
    'Peach',
    'Banana',
    'Beetroot',
    'Blueberry',
    'Cocos',
    'Corn Husk',
    'Ginger Root',
    'Grape Blue',
    'Papaya',
    'Nut Pecan',
    'Potato Red'
]

bird_labels = [
    'Dalmatian Pelican',
    'Black Breasted Puffbird',
    'Wattled Curassow',
    'American Wigeon',
    'Carmine Bee-Eater',
    'Gambels Quail',
    'Umbrella Bird',
    'American Kestrel',
    'American Goldfinch',
    'Blue Grosbeak',
    'Masked Lapwing',
    'Laughing Gull',
    'American Robin',
    'Malabar Hornbill',
    'Ocellated Turkey',
    'Peregrine Falcon',
    'Squacco Heron',
    'Chinese Bamboo Partridge',
    'Ibisbill',
    'Rosy Faced Lovebird',
    'Greator Sage Grouse',
    'Philippine Eagle',
    'Red Fody',
    'Anhinga',
    'Brandt Cormarant',
    'Bird of Paradise',
    'Lilac Roller',
    'Eastern Golden Weaver',
    'Bald Eagle',
    'Kiwi'
]

"""## Métodos auxiliares para pre-processamento dos dados

Nesta etapa do projeto, serão incluídos dois métodos importantes para o pré-processamento dos dados dos datasets: `prepareDirForFilteredData` e `filterData`. Esses métodos visam garantir a consistência e a organização dos dados utilizados no projeto.

**prepareDirForFilteredData**: tem como objetivo realizar a preparação do diretório para os dados filtrados. Essa etapa é importante para garantir a idempotência da execução, ou seja, garantir que sempre tenhamos um diretório limpo e organizado para armazenar os dados filtrados, independentemente de quantas vezes o código seja executado.

**filterData**: é responsável por filtrar os dados das classes desejadas. Utilizando as labels definidas anteriormente, esse método copia todas as informações do diretório fonte para o diretório filtrado. O diretório filtrado é o local onde serão armazenados apenas os dados relevantes que serão utilizados no projeto.
"""

import os
import shutil

def prepareDirForFilteredData(filteredTrainDataDir, filteredValidationDataDir, filteredTestDataDir):
  # Wipe train/test directories to ensure idempotency of execution
  shutil.rmtree(filteredTrainDataDir, ignore_errors=True)
  shutil.rmtree(filteredValidationDataDir, ignore_errors=True)
  shutil.rmtree(filteredTestDataDir, ignore_errors=True)

  # Create train/test directories for execution
  os.makedirs(filteredTrainDataDir, exist_ok=True)
  os.makedirs(filteredValidationDataDir, exist_ok=True)
  os.makedirs(filteredTestDataDir, exist_ok=True)

# labelManipulation is needed becasue each datase has different patterns for folder namin. Expected values [upper, title]
def filterData(sourceDir, targetDir, labels, labelManipulation):
  for label in labels:

    src_dir = os.path.join(sourceDir, getattr(label, labelManipulation)()) # get absolute dir for source
    dest_dir = os.path.join(targetDir, getattr(label, labelManipulation)()) # get absolute dir for target

    files = os.listdir(src_dir) # get all files under source dir
    os.makedirs(dest_dir, exist_ok=True) # create subfoler from <sourceDir> in <targetDir> for copy

    # for all files matching <labels> from <sourceDir> copies to <targetDir>
    for file in files:
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)

        shutil.copy(src_file, dest_file)

"""Esses dois métodos são essenciais para garantir que apenas os dados relevantes sejam utilizados no projeto, facilitando a organização e a execução das etapas seguintes do pipeline de aprendizado de máquina.

Em seguida, aplicamos estas funções para ambos os diretórios.
"""

import keras
from keras.preprocessing.image import ImageDataGenerator

birds_train_data_dir = 'birdies/train'
birds_validation_data_dir = 'birdies/valid'
birds_test_data_dir = 'birdies/test'

birds_filtered_train_data_dir = 'filtered/birds/train'
birds_filtered_validation_data_dir = 'filtered/birds/valid'
birds_filtered_test_data_dir = 'filtered/birds/test'

# Create directories for filtered data
prepareDirForFilteredData(birds_filtered_train_data_dir, birds_filtered_validation_data_dir, birds_filtered_test_data_dir)

# Copy data from <birds_train_data_dir> to <birds_filtered_train_data_dir> matching <bird_labels>
filterData(birds_train_data_dir, birds_filtered_train_data_dir, bird_labels, "upper")

# Copy data from <birds_validation_data_dir> to <birds_filtered_validation_data_dir> matching <bird_labels>
filterData(birds_validation_data_dir, birds_filtered_validation_data_dir, bird_labels, "upper")

# Copy data from <birds_test_data_dir> to <birds_filtered_test_data_dir> matching <bird_labels>
filterData(birds_test_data_dir, birds_filtered_test_data_dir, bird_labels, "upper")

import keras
from keras.preprocessing.image import ImageDataGenerator

fruits_train_data_dir = 'fruits/fruits-360_dataset/fruits-360/Training'
fruits_validation_data_dir = 'fruits/fruits-360_dataset/fruits-360/Test'

fruits_filtered_train_data_dir = 'filtered/fruits/train'
fruits_filtered_validation_data_dir = 'filtered/fruits/valid'
fruits_filtered_test_data_dir = 'filtered/fruits/test'

# Create directories for filtered data
prepareDirForFilteredData(fruits_filtered_train_data_dir, fruits_filtered_validation_data_dir, fruits_filtered_test_data_dir)

# Copy data from <birds_train_data_dir> to <birds_filtered_train_data_dir> matching <bird_labels>
filterData(fruits_train_data_dir, fruits_filtered_train_data_dir, fruit_labels, "title")

# Copy data from <birds_validation_data_dir> to <birds_filtered_test_data_dir> matching <bird_labels>
filterData(fruits_validation_data_dir, fruits_filtered_validation_data_dir, fruit_labels, "title")

# Copy data from <birds_validation_data_dir> to <birds_filtered_test_data_dir> matching <bird_labels>
filterData(fruits_validation_data_dir, fruits_filtered_test_data_dir, fruit_labels, "title")

"""### Pré-processamento e geração dos dados

Nesse trecho de código, estamos configurando o pré-processamento dos dados e criando geradores de dados para treinamento, validação e teste da CNN.

1. **Redimensionamento:**

 Ajustar o tamanho das imagens para garantir que todas possuem o mesmo tamanho - assim tenho um tamanho consistente e menor, reduzindo os requisitos computacionais necessários


1. **Definição das dimensões das imagens:**

  img_width e img_height representam as dimensões desejadas para redimensionar as imagens de entrada. Nesse caso, as imagens serão redimensionadas para terem 100 pixels de largura e 100 pixels de altura.

2. **Definição do tamanho do lote (batch size):**

  batch_size representa o número de amostras que serão propagadas pela rede neural de uma vez durante o treinamento. Nesse caso, estamos definindo o tamanho do lote como 32, o que significa que serão processadas 32 imagens de uma vez.

1. **Configuração do pré-processamento das imagens:**

  Utilizamos a classe ImageDataGenerator da biblioteca Keras para realizar o pré-processamento das imagens. Aqui, fazemos a normalização de nossas imagens, na qual dividiremos o valor de cada pixel por 255 , o que os dimensiona para a faixa de [0, 1]. A normalização ajuda a estabilizar o processo de treinamento e garante que todos os recursos contribuam igualmente.

5. **Criação dos geradores de dados:**

  Utilizamos o método flow_from_directory dos objetos ImageDataGenerator para criar geradores de dados para treinamento, validação e teste.
  Para cada conjunto de dados (treinamento, validação e teste), especificamos o diretório onde as imagens estão armazenadas (birds_filtered_train_data_dir, birds_filtered_validation_data_dir e birds_filtered_test_data_dir).
  Especificamos também o tamanho das imagens de entrada (target_size) e o tamanho do lote (batch_size) que definimos anteriormente.
  Por fim, definimos class_mode como 'categorical' para indicar que estamos trabalhando com classificação multiclasse.

7. **Obtendo o número de classes:**

  No final do trecho de código, utilizamos len(bird_labels) para obter o número de classes presentes no conjunto de dados. Isso será útil para definir o número de unidades na camada de saída da nossa CNN.

Essas etapas são essenciais para preparar os dados de treinamento, validação e teste da CNN, garantindo que as imagens sejam pré-processadas corretamente e que os geradores de dados forneçam lotes de imagens para o treinamento do modelo.

#### Processamento de dados para Birds Dataset
"""

img_width, img_height = 100, 100 # according to values defined in pre-processing
batch_size = 32 # validate colab performance eventually

# Divide pixel by 255, resulting in a value between 0 and 1
birds_train_datagen = ImageDataGenerator(rescale=1./255)
birds_validation_datagen = ImageDataGenerator(rescale=1./255)
birds_test_datagen = ImageDataGenerator(rescale=1./255)

# Generate data for training
birds_train_generator = birds_train_datagen.flow_from_directory(
    birds_filtered_train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Generate data for validation
birds_validation_generator = birds_validation_datagen.flow_from_directory(
    birds_filtered_validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Generate data for test
birds_test_generator = birds_test_datagen.flow_from_directory(
    birds_filtered_test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

len(bird_labels)

"""#### Processamento de dados para Fruits Dataset"""

img_width, img_height = 100, 100 #como definido na sessão de pre-processamento
batch_size = 32 #verificar como fica com o consumo de memória no colab

# Divide pixel by 255, resulting in a value between 0 and 1
fruits_train_datagen = ImageDataGenerator(rescale=1./255)
fruits_validation_datagen = ImageDataGenerator(rescale=1./255)
fruits_test_datagen = ImageDataGenerator(rescale=1./255)

# Generate data for training
fruits_train_generator = fruits_train_datagen.flow_from_directory(
    fruits_filtered_train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Generate data for validation
fruits_validation_generator = fruits_validation_datagen.flow_from_directory(
    fruits_filtered_validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Generate data for test
fruits_test_generator = birds_test_datagen.flow_from_directory(
    fruits_filtered_test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

len(fruit_labels)

"""#  Treinamento

Antes de descrever a arquitetura da rede utilizada e o processo de treinamento, é importante destacar a importância dessas etapas no desenvolvimento da nossa solução. A escolha da arquitetura da rede e a definição dos parâmetros de treinamento são fundamentais para alcançar um desempenho satisfatório na tarefa de classificação de imagens. Além disso, o processo de treinamento envolve a otimização dos pesos da rede por meio da minimização da função de erro, visando encontrar os melhores valores que permitam a rede generalizar bem para novos exemplos. Portanto, vamos explorar detalhadamente a arquitetura da nossa CNN e as estratégias de treinamento adotadas para obter resultados consistentes e precisos.

## Rede utilizada

Vamos agora abordar a rede utilizaremos para resolver o problema de classificação de imagens, abordando a importância de cada parâmetro para essa definição:

</br>

**Camadas convolucionais:**

  As camadas `Conv2D` são responsáveis por extrair características das imagens. O primeiro parâmetro indica o número de filtros que serão aplicados, o segundo parâmetro define o tamanho desses filtros (3x3) e a função de ativação `relu` é usada para introduzir não linearidade nas saídas das camadas convolucionais.

</br>

**Camadas de pooling:**

  As camadas `MaxPooling2D `são usadas para reduzir a dimensionalidade das características extraídas pelas camadas convolucionais. O parâmetro `pool_size` define o tamanho da janela de pooling.

</br>

**Camada de flatten:**

  A camada Flatten é usada para converter o volume de características em um vetor unidimensional, preparando os dados para a entrada nas camadas densas.

</br>

**Camadas densas:**

  As camadas Dense são camadas totalmente conectadas, onde cada neurônio recebe entradas de todos os neurônios da camada anterior. O parâmetro 256 indica o número de neurônios nessa camada, e a função de ativação `relu` é usada para introduzir não linearidade nas saídas.
  A camada Dropout é usada para regularizar o modelo, desativando aleatoriamente um percentual (0.4) dos neurônios durante o treinamento, com o objetivo de evitar overfitting.

</br>

**Camada de saída:**

  A última camada Dense tem um número de neurônios igual ao número de classes presentes no conjunto de dado ou seja `len(bird_labels)` e `len(fruit_labels)`. A função de ativação `softmax` é usada para obter uma distribuição de probabilidade sobre as classes, indicando a probabilidade de cada classe ser a classe correta.

</br>

**Compilação:**
  Definimos o otimizador como `adam`, que é um algoritmo de otimização popular para redes neurais. A função de perda é definida como `categorical_crossentropy`, adequada para problemas de classificação multiclasse. Também especificamos a métrica de avaliação como `accuracy`, que mede a acurácia do modelo durante o treinamento e teste.

Cada parâmetro da arquitetura da rede desempenha um papel importante na aprendizagem e no desempenho da CNN. O ajuste adequado desses parâmetros é essencial para obter resultados satisfatórios na tarefa de classificação de imagens.

## Métodos de Treinamento e Validação
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def trainModel(datasetGenerator, datasetValidator, labels, numberConvLayers, neuronBaseSize, activationFunction, errorFunction, optimizer, numberEpochs, batchSize):
  model = Sequential()

  for layer in range(numberConvLayers + 1):
    model.add(Conv2D(neuronBaseSize * (layer + 1), (3, 3), activation=activationFunction, input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(neuronBaseSize * (numberConvLayers + 1), activation=activationFunction))
  model.add(Dropout(0.4))
  model.add(Dense(len(labels), activation='softmax'))

  model.compile(optimizer=optimizer, loss=errorFunction, metrics=['accuracy'])

  # cria um checkpoint para salvar os pesos do melhor modelo encontrado no trainamento
  checkpointer = ModelCheckpoint(filepath='model.weights.best.h5', verbose=0, save_best_only=True)

  modelFit = model.fit(
    datasetGenerator,
    steps_per_epoch=datasetGenerator.samples // batchSize,
    validation_data=datasetValidator,
    validation_steps=datasetValidator.samples // batchSize,
    callbacks=[checkpointer],
    epochs=numberEpochs)

  plt.plot(modelFit.history['loss'])
  plt.plot(modelFit.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper right')
  plt.show()

  plt.plot(modelFit.history['accuracy'])
  plt.plot(modelFit.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  return model

def measureAccuracy(testGenerator, experimentModel):
  num_samples = testGenerator.samples # Get the total number of samples in the test set
  ground_truth_labels = [] # Get the class labels for the test set in order

  # Iterate over each batch of data from the generator
  for i in range(len(testGenerator)):
      # Get the batch of images and their corresponding labels
      _ , batch_labels = testGenerator[i]

      # Append the batch labels to the ground truth labels list
      ground_truth_labels.extend(np.argmax(batch_labels, axis=1))

  # Convert the ground truth labels to a numpy array
  ground_truth_labels = np.array(ground_truth_labels)

  # Initialize an empty array to store the predicted labels
  predicted_labels = np.empty(shape=(num_samples,), dtype=int)

  # Iterate over each batch of data from the generator again
  for i in range(len(testGenerator)):
      batch_images, _ = testGenerator[i] # Get the batch of images
      batch_predictions = experimentModel.predict(batch_images) # Make predictions for the batch

      batch_predicted_labels = np.argmax(batch_predictions, axis=1) # Convert the predictions to class labels

      predicted_labels[i * batch_size : (i+1) * batch_size] = batch_predicted_labels # Store the predicted labels for the batch in the final array

  # Compare the predicted labels with the ground truth labels
  correct_predictions = np.sum(predicted_labels == ground_truth_labels)
  total_predictions = len(ground_truth_labels)

  # Calculate accuracy
  accuracy = correct_predictions / total_predictions

  print('Accuracy:', accuracy)

"""## Treinamento do modelo para Birds Dataset"""

print("Training first experiment")

experimentOne = trainModel(
    datasetGenerator= birds_train_generator,
    datasetValidator = birds_validation_generator,
    labels= bird_labels,
    numberConvLayers= 3,
    neuronBaseSize= 16,
    activationFunction= 'relu',
    errorFunction= 'categorical_crossentropy',
    optimizer= 'adam',
    numberEpochs= 10,
    batchSize= batch_size
)

print("Training second experiment")

experimentTwo = trainModel(
    datasetGenerator= birds_train_generator,
    datasetValidator = birds_validation_generator,
    labels= bird_labels,
    numberConvLayers= 3,
    neuronBaseSize= 32,
    activationFunction= 'relu',
    errorFunction= 'categorical_crossentropy',
    optimizer= 'adam',
    numberEpochs= 10,
    batchSize= batch_size
)

print("Training third experiment")

experimentThree = trainModel(
    datasetGenerator= birds_train_generator,
    datasetValidator = birds_validation_generator,
    labels= bird_labels,
    numberConvLayers= 4,
    neuronBaseSize= 32,
    activationFunction= 'relu',
    errorFunction= 'categorical_crossentropy',
    optimizer= 'adam',
    numberEpochs= 10,
    batchSize= batch_size
)

"""###**Resultados**"""

print("Acuracy experiment one:")
measureAccuracy(birds_test_generator, experimentOne)

print("-------------------------------------------------------------")
print("Acuracy experiment two:")

measureAccuracy(birds_test_generator, experimentTwo)
print("-------------------------------------------------------------")
print("Acuracy experiment three:")

measureAccuracy(birds_test_generator, experimentThree)

"""## Treinamento modelo para Fruits Dataset"""

print("Training first experiment")

fruitsExperimentOne = trainModel(
    datasetGenerator= fruits_train_generator,
    datasetValidator = fruits_validation_generator,
    labels= fruit_labels,
    numberConvLayers= 3,
    neuronBaseSize= 8,
    activationFunction= 'relu',
    errorFunction= 'categorical_crossentropy',
    optimizer= 'adam',
    numberEpochs= 5,
    batchSize= batch_size
)

print("Training second experiment")

fruitsExperimentTwo = trainModel(
    datasetGenerator= fruits_train_generator,
    datasetValidator = fruits_validation_generator,
    labels= fruit_labels,
    numberConvLayers= 3,
    neuronBaseSize= 16,
    activationFunction= 'relu',
    errorFunction= 'categorical_crossentropy',
    optimizer= 'adam',
    numberEpochs= 5,
    batchSize= batch_size
)

print("Training third experiment")

fruitsExperimentThree = trainModel(
    datasetGenerator= fruits_train_generator,
    datasetValidator = fruits_validation_generator,
    labels= fruit_labels,
    numberConvLayers= 3,
    neuronBaseSize= 32,
    activationFunction= 'relu',
    errorFunction= 'categorical_crossentropy',
    optimizer= 'adam',
    numberEpochs= 5,
    batchSize= batch_size
)

print("Acuracy experiment one:")
measureAccuracy(fruits_test_generator, fruitsExperimentOne)

print("-------------------------------------------------------------")
print("Acuracy experiment two:")

measureAccuracy(fruits_test_generator, fruitsExperimentTwo)
print("-------------------------------------------------------------")
print("Acuracy experiment three:")

measureAccuracy(fruits_test_generator, fruitsExperimentThree)

"""###**Resultados**"""

print("Acuracy experiment one:")
measureAccuracy(fruits_test_generator, experimentOne)

fruitsExperimentOne.load_weights('model.weights.best.h5')
score = experimentOne.evaluate(fruits_test_generator)
print('\n', 'Acurácia:', score[1])

print("-------------------------------------------------------------")
print("Acuracy experiment two:")

measureAccuracy(fruits_test_generator, experimentTwo)

fruitsExperimentTwo.load_weights('model.weights.best.h5')
score = experimentTwo.evaluate(fruits_test_generator)
print('\n', 'Acurácia:', score[1])

print("-------------------------------------------------------------")
print("Acuracy experiment three:")

measureAccuracy(fruits_test_generator, experimentThree)

fruitsExperimentThree.load_weights('model.weights.best.h5')
score = experimentThree.evaluate(fruits_test_generator)
print('\n', 'Acurácia:', score[1])

"""# Analise de Resultados

## Birds 525

O treinamento da rede neural utilizando o dataset BIRDS 525 SPECIES do Kaggle resultou em uma acurácia de ~0.82666665. Essa acurácia indica a proporção de previsões corretas feitas pela rede neural em relação ao total de exemplos de treinamento utilizados.

Uma acurácia de ~0.82666665, ou seja ~82%, é considerada relativamente alta e sugere que o modelo de rede neural obteve um bom desempenho na tarefa de classificação das espécies de aves selecionadas. Isso significa que a rede neural foi capaz de aprender padrões e características distintivas nas imagens das aves, permitindo que ela fizesse previsões com uma taxa de acerto significativa. Uma acurária como esta é um bom resultado para um modelo de rede neural treinado em um dataset desafiador como o BIRDS 525 SPECIES.

Em relação ao pré-processamento, analisamos conforme o desenvolvimento da rede neural, que a "Remoção do Fundo" e o "Tratamento do desequilíbrio de classes", como operações prévias, não foram necessárias, pois o resultado final já foi satisfatório. O principal motivo para evitar a utilização destes pré-processamentos tem a ver com o custo benefício, filtrar e processar digitalmente cada imagem ".jpg" pode ser um trabalho bem demorado para executar, portanto para datasets mais complexos, o pré-processamento teria mais custo benefício, no nosso caso não foi necessário.

## Fruits 360

Por outro lado, o treinamento da rede neural, utilizando o dataset FRUITS 360 do Kaggle, atingiu 0.984631, ou 98,46%, de acurácia. Isto indica que a rede possui um ótimo desempenho quando se trata da classificação das classes escolhidas para realização do treinamento. Também vale ressaltar que este valor alto de acurácia já aparece após primeira época, e isto pode se dar pela qualidade das imagens e pela capacidade da rede de aprender padrões e distinções em imagens de frutas, possuindo um alto percentual de acerto.

Aplicando os mesmos parâmetros de treinamento, tivemos um aumento significativo na precisão da clissificação, quando comparamos o treinamento feito com o BIRDS 525 SPECIES e o FRUITS 360. Isto é devido a qualidade das imagens, sendo as frutas sem fundo, facillitando a classificação para o modelo. Fazendo uma rápida comparação com outros resultados de treinamentos feitos neste mesmo datasest, a acurácia de 98% é o esperado em redes com bom desempenho.

# Conclusão

Em conclusão, o processamento das redes neurais nos conjuntos de dados BIRDS 525 SPECIES e FRUITS 360 resultou em acurácias distintas, refletindo o desempenho de cada modelo nos respectivos problemas de classificação.

No caso do dataset BIRDS 525 SPECIES, a rede neural obteve uma acurácia de 0.82666665. Esse resultado indica que o modelo foi capaz de aprender e reconhecer padrões relevantes nas imagens das aves, alcançando um nível de precisão considerável.

Já no dataset FRUITS 360, a rede neural alcançou uma acurácia mais elevada, atingindo 0.984631. Essa alta acurácia sugere que o modelo foi capaz de aprender de forma eficaz as características distintivas das frutas presentes no conjunto de dados. Essa precisão elevada é um indicativo do bom desempenho da rede neural no reconhecimento e classificação das diferentes frutas.

Os resultados obtidos nas redes neurais treinadas nos conjuntos de dados BIRDS 525 SPECIES e FRUITS 360 demonstram o potencial dessas técnicas em problemas de classificação. Enquanto a acurácia de 0.82666665 no primeiro dataset mostra um bom desempenho considerando a complexidade das espécies de aves, a acurácia de 0.984631 no segundo dataset destaca a capacidade de reconhecimento das diferentes frutas, que em comparação com aves, possuem características mais simples e facilmente identificáveis. Esses resultados ressaltam a eficácia das redes neurais e a importância de selecionar conjuntos de dados adequados e aplicar métricas abrangentes para uma análise completa do desempenho.
"""