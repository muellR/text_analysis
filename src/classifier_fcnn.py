# hier sollen die Inputdaten in einem Modell gelern werden:
# 1. Embedden des Text
# 2. lernen eines Neuronalen Nezt (FCCN)
# 3. Vorhersage und Zusammenfassung (Konfusionsmatrix)
#   Option: Zusätliches Lernen mit abgeleitetem Sentiment und Vergleich der beiden outcomes

import base
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') # "Anti-Grain Geometry" → rendert nur in den Speicher (RAM)
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

# BAAI/bge-base-en-v1.5
# all-MiniLM-L12-v2
if 1==1:
    s_sentence_transformer='BAAI/bge-base-en-v1.5'
    s_senctence_tranformer_short = '_bge'
else:
    s_sentence_transformer='all-MiniLM-L12-v2'
    s_senctence_tranformer_short = '_mini'



# Daten laden und splitten --> 2 Varianten: mit oder ohne Sentiment als Feature
train_df, test_df = pd, pd
if 1==1:
    s_praefix = "class_fcnn" + s_senctence_tranformer_short
    train_df, test_df = base.load_and_split_data(s_file_name='fake reviews dataset.csv', 
                                                 b_encode_label=True, 
                                                 b_add_embeddings=True, 
                                                 b_normalize_rating=True, 
                                                 b_one_hot_category=True, 
                                                 s_sentence_transformer=s_sentence_transformer
                                            )
else:
    s_praefix = "class_fcnn_with_sentiment" + s_senctence_tranformer_short
    train_df, test_df = base.load_and_split_data(s_file_name='fake reviews dataset_senti.csv', 
                                                 b_encode_label=True, 
                                                 b_add_embeddings=True, 
                                                 b_normalize_rating=True, 
                                                 b_one_hot_category=True, 
                                                 s_sentence_transformer=s_sentence_transformer
                                                 )

# BAAI/bge-base-en-v1.5
# all-MiniLM-L12-v2

print(train_df.shape)
print(train_df.columns)

print(test_df.shape)
print(test_df.columns)



# EarlyStopping Callback definieren
early_stopping = EarlyStopping(
    monitor='val_loss',         # Metrik beobachten
    patience=5,                 # Epochen warten bevor abgebrochen wird
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
    col == 'rating_norm' or
    col == 'sentiment' or
    col == 'accuracy'
]


# die Features und Labels in numpy arrays umwandeln
X_train = train_df[feature_cols].values
Y_train = train_df['label_encoded'].values

X_val = test_df[feature_cols].values
Y_val = test_df['label_encoded'].values



# TEST standardisieren der Roh Features
# nicht nötig bei SBERT Embeddings, da diese bereits normalisiert sind, aber könnte bei anderen Features helfen
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)



# fully connected neural network with moderate dropout
input = keras.Input(shape=(X_train.shape[1],))

# x = Dense(512, activation='gelu')(input)
# x = Dropout(0.2)(x) # weniger Dropout bei SBERT
x = Dense(256, activation='gelu')(input) # gelu > relu für Embeddings
x = BatchNormalization()(x)
x = Dropout(0.2)(x) # weniger Dropout bei SBERT
x = Dense(128, activation='gelu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)



output = Dense(1, activation='sigmoid')(x)
class_fcnn = keras.Model(input, output)

# compile model and initialize weights
adam = optimizers.Adam()
class_fcnn.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])



# summarize the model
class_fcnn.summary()



# train the model
if b_init or not os.path.exists(os.path.join(base.models_dir, s_praefix + '.keras')):
    
    history = class_fcnn.fit(X_train, Y_train,
        batch_size=512,
        epochs=50,
        verbose=0,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping, checkpoint] # use EarlyStopping callback to prevent overfitting and save best model
        )

    # Model / history speichern
    class_fcnn.save(os.path.join(base.models_dir, s_praefix + '.keras'))
    with open(os.path.join(base.models_dir, s_praefix + '_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
else:
    # load model from file
    class_fcnn = keras.models.load_model(os.path.join(base.models_dir, s_praefix + '.keras'))
    with open(os.path.join(base.models_dir, s_praefix + '_history.pkl'), 'rb') as f:
        history = pickle.load(f)



# Lernkurven plotten
fig =base.plot_learning_curves(history);
fig.suptitle(s_praefix, fontsize=14, fontweight='bold', y=1.02)
base.reinitialize_folders(base.figures_dir)
fig.savefig(os.path.join(base.figures_dir, s_praefix + '_learning_curves.png'), dpi=150, bbox_inches='tight')




# Evaluation auf Testdaten - predict each instance of the testset
pred = class_fcnn.predict(X_val)
pred_classes = (pred > 0.5).astype(int)

pred_classes = np.ravel(pred_classes) # flatten to 1D array for comparison with Y_val

# classification report F1, precision, recall
report = classification_report(Y_val, pred_classes)
print("Classification Report:\n", report)



# test_df auf die Originalspalten reduzieren
save_df = test_df[['category', 'rating', 'label', 'text_', 'id', 'label_encoded']]

# Falsch und korrekt klassifizierte Datensätze je in ein File schreiben
correct_mask = (pred_classes == Y_val)
wrong_mask = ~correct_mask # invert mask to get wrong predictions
correct_df = save_df.loc[correct_mask].sort_values("id")
wrong_df = save_df.loc[wrong_mask].sort_values("id")
base.reinitialize_folders(base.predictions_dir, drop_existing=False)
correct_df.to_csv(os.path.join(base.predictions_dir, "correct_predictions_" + s_praefix + ".csv"), index=False)
wrong_df.to_csv(os.path.join(base.predictions_dir, "wrong_predictions_" + s_praefix + ".csv"), index=False)
print("Path to predictions:", base.predictions_dir)






# get confusion matrix
cm = confusion_matrix(Y_val, pred_classes)

acc_fc = np.mean(Y_val == pred_classes.flatten())
print("Accuracy = ", acc_fc)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='viridis')
plt.title('Confusion Matrix ' + s_praefix)
plt.savefig(os.path.join(base.figures_dir, s_praefix + '_confusion_matrix.png'), dpi=150, bbox_inches='tight')






# Kennzahlen aus den Lernkurven extrahieren
min_val_loss = min(history.history['val_loss'])
print("Min Validation Loss:", min_val_loss)
max_val_acc = max(history.history['val_accuracy'])
print("Max Validation Accuracy:", max_val_acc)