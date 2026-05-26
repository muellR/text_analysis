# hier sollen die Inputdaten in einem Modell gelern werden:
# 1. Embedden des Text
# 2. lernen eines Neuronalen Nezt (FCCN)
# 3. Vorhersage und Zusammenfassung (Konfusionsmatrix)
#   Option: Zusätliches Lernen mit abgeleitetem Sentiment und Vergleich der beiden outcomes

import base
import json
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('default')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Keras Building blocks
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

b_init = True # if false, do not force Model regeneration, but load existing model if it exists
b_early_stopping = True # if true, use EarlyStopping callback to prevent overfitting and save best model

# Daten laden und splitten --> 2 Varianten: mit oder ohne Sentiment als Feature
train_df, test_df = pd, pd
if 1==2:
    s_praefix = "model_02"
    train_df, test_df = base.load_and_split_data(s_file_name='fake reviews dataset.csv')
else:
    s_praefix = "model_02_with_sentiment"
    train_df, test_df = base.load_and_split_data(s_file_name='fake reviews dataset_senti.csv')

if b_early_stopping:
    s_praefix += '_early_stopping'


print(train_df.shape)
print(train_df.columns)



# EarlyStopping Callback definieren
early_stopping = EarlyStopping(
    monitor='val_loss',      # Metrik beobachten
    patience=5,              # Epochen warten bevor abgebrochen wird
    restore_best_weights=True,  # Bestes Modell wiederherstellen
    verbose=1
)

# Optional: Bestes Modell zwischenspeichern
checkpoint = ModelCheckpoint(
    filepath=os.path.join(base.models_dir, s_praefix + '.keras'),
    monitor='val_loss', # 'val_loss' oder 'val_accuracy'
    save_best_only=True,
    verbose=1
)


# features wählen
feature_cols = [col for col in train_df.columns if
    col.startswith('emb_') or
    col.startswith('category_') or
    col == 'rating' or
    col == 'sentiment' or
    col == 'accuracy'
]

# die Features und Labels in numpy arrays umwandeln
X_train = train_df[feature_cols].values
Y_train = train_df['label'].values

X_val = test_df[feature_cols].values
Y_val = test_df['label'].values


# fully connected neural network with moderate dropout
input = keras.Input(shape=(X_train.shape[1],))

# x = Dense(512, activation='relu')(input)
# x = Dropout(0.2)(x) # weniger Dropout bei SBERT
x = Dense(256, activation='gelu')(input) # gelu > relu für Embeddings
x = BatchNormalization()(x)
x = Dropout(0.2)(x) # weniger Dropout bei SBERT
x = Dense(128, activation='gelu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='sigmoid')(x)
model_02 = keras.Model(input, output)

# compile model and initialize weights
adam = optimizers.Adam()
model_02.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



# summarize the model
model_02.summary()



# train the model
if b_init or not os.path.exists(os.path.join(base.models_dir, s_praefix + '.keras')):
    
    if b_early_stopping:
        history = model_02.fit(X_train, Y_train,
            batch_size=512,
            epochs=50,
            verbose=0,
            validation_data=(X_val, Y_val),
            callbacks=[early_stopping, checkpoint]
            )
    else:
        history = model_02.fit(X_train, Y_train,
            batch_size=512,
            epochs=50,
            verbose=0,
            validation_data=(X_val, Y_val)
            )
 
    # Model / history speichern
    model_02.save(os.path.join(base.models_dir, s_praefix + '.keras'))
    with open(os.path.join(base.models_dir, s_praefix + '_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
else:
    # load model from file
    model_02 = keras.models.load_model(os.path.join(base.models_dir, s_praefix + '.keras'))
    with open(os.path.join(base.models_dir, s_praefix + '_history.pkl'), 'rb') as f:
        history = pickle.load(f)



# Lernkurven plotten
fig =base.plot_learning_curves(history);
fig.suptitle(s_praefix, fontsize=14, fontweight='bold', y=1.02)
base.reinitialize_folders(base.figures_dir)
fig.savefig(os.path.join(base.figures_dir, s_praefix + '_learning_curves.png'), dpi=150, bbox_inches='tight')






# Auswertung des Modells auf dem Testset
# predict each instance of the testset
pred = model_02.predict(X_val)
pred_classes = (pred > 0.5).astype(int)

# get confusion matrix
cm = confusion_matrix(Y_val, pred_classes)

acc_fc = np.mean(Y_val == pred_classes.flatten())
print("Accuracy = ", acc_fc)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='viridis')
plt.title('Confusion Matrix ' + s_praefix)
plt.savefig(os.path.join(base.figures_dir, s_praefix + '_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()


# classification report F1, precision, recall
print(classification_report(Y_val, pred_classes))




# Kennzahlen aus den Lernkurven extrahieren
min_val_loss = min(history.history['val_loss'])
print("Min Validation Loss:", min_val_loss)
max_val_acc = max(history.history['val_accuracy'])
print("Max Validation Accuracy:", max_val_acc)