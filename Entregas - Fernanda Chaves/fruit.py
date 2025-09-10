# ================================
# 1. Preparação do ambiente
# ================================
!pip install -q kaggle tensorflow matplotlib seaborn pillow scikit-learn

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

# ================================
# 2. Download do dataset (Kaggle)
# ================================
# -> Antes, faça upload do arquivo kaggle.json no Colab (credenciais da API)
from google.colab import files
files.upload()   # selecione o kaggle.json baixado do Kaggle

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Baixar dataset Fresh & Rotten Fruits
!kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification
!unzip -q fruits-fresh-and-rotten-for-classification.zip -d fruits_data

# ================================
# 3. Carregamento dos dados
# ================================
base_dir = "fruits_data/fruits-fresh-and-rotten-for-classification"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalização (0–1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Classes
nomes_classes = train_ds.class_names
print("Classes:", nomes_classes)

# ================================
# 4. Construção do Modelo (CNN)
# ================================
modelo_food = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(nomes_classes), activation="softmax")
])

modelo_food.summary()

# ================================
# 5. Compilação e Treinamento
# ================================
modelo_food.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

history = modelo_food.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5
)

# ================================
# 6. Gráficos de Acurácia e Perda
# ================================
plt.figure(figsize=(12,5))

# Acurácia
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Treino")
plt.plot(history.history['val_accuracy'], label="Validação")
plt.title("Evolução da Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()

# Perda
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Treino")
plt.plot(history.history['val_loss'], label="Validação")
plt.title("Evolução da Perda")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ================================
# 7. Avaliação Final
# ================================
erro_teste, acc_teste = modelo_food.evaluate(test_ds, verbose=2)
print("\n✅ Acurácia com dados de Teste:", acc_teste)

# Matriz de confusão
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = np.argmax(modelo_food.predict(test_ds), axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=nomes_classes,
            yticklabels=nomes_classes)
plt.title("Matriz de Confusão - Fresh vs Rotten Fruits")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# ================================
# 8. Teste com Nova Imagem
# ================================
# Troque o caminho pela sua imagem (ex: "banana_rotten.jpg")
nova_imagem = Image.open("/content/banana_rotten.jpg").resize(IMG_SIZE)
plt.imshow(nova_imagem)
plt.axis("off")
plt.show()

nova_array = np.expand_dims(np.array(nova_imagem)/255.0, axis=0)
previsao = modelo_food.predict(nova_array)
classe_prevista = np.argmax(previsao)
print("🍌 A nova imagem foi classificada como:", nomes_classes[classe_prevista])
