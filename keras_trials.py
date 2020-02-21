#1190000
# build ml model with keras
model = keras.Sequential()
model.add(keras.layers.Dense(422, input_dim=422, kernel_initializer='normal', activation='relu')) #input dimensions must be == to number of features
model.add(keras.layers.Dense(150, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(50, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(20, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(10, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=.0005))
model.fit(X, Y, epochs=200, batch_size=100, verbose=2, shuffle=True) #epochs: how many times to run through, batch_size:how sets of data points to train on per epoch, verbose: how training progress is shown
model.save('NYC_apartment_price.h5') # save ml model

#1180000
# build ml model with keras
model = keras.Sequential()
model.add(keras.layers.Dense(422, input_dim=422, kernel_initializer='normal', activation='relu')) #input dimensions must be == to number of features
model.add(keras.layers.Dense(150, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(50, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(50, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(20, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(10, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=.0008))
model.fit(X, Y, epochs=200, batch_size=100, verbose=2, shuffle=True) #epochs: how many times to run through, batch_size:how sets of data points to train on per epoch, verbose: how training progress is shown
model.save('NYC_apartment_price.h5') # save ml model

#1200000
# build ml model with keras
model = keras.Sequential()
model.add(keras.layers.Dense(422, input_dim=422, kernel_initializer='normal', activation='relu')) #input dimensions must be == to number of features
model.add(keras.layers.Dense(250, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=.00005))
model.fit(X, Y, epochs=50, batch_size=200, verbose=2, shuffle=True) #epochs: how many times to run through, batch_size:how sets of data points to train on per epoch, verbose: how training progress is shown
model.save('NYC_apartment_price.h5') # save ml model
