import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential

def main():

    # training & test data
    delta30 = np.loadtxt('data_fix/delta30.csv',delimiter=',')
    dwdm0 = np.loadtxt('data_fix/dwdm0.csv',delimiter=',')
    dwdm1 = np.loadtxt('data_fix/dwdm1.csv',delimiter=',')
    fyprofile = np.loadtxt('data_fix/fyprofile.csv',delimiter=',')
    delta = np.stack((delta30,delta30),axis=2)
    dwdm = np.stack((dwdm0, dwdm1),axis=2)

    x_data = np.stack((delta,dwdm),axis=3)
    y_data = fyprofile

    i_train = np.loadtxt('data_fix/i_train.csv',delimiter=',')
    i_val = np.loadtxt('data_fix/i_val.csv',delimiter=',')
    i_train = np.asarray(np.hstack((i_train, i_val))-1,dtype=np.int)
    i_test = np.asarray(np.loadtxt('data_fix/i_test.csv',delimiter=',')-1,dtype=np.int)

    x_train = x_data[i_train,:,:,:]
    y_train = y_data[i_train,:]
    x_test = x_data[i_test,:,:,:]
    y_test = y_data[i_test,:]

    # define model
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(4, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.025))
    model.add(Conv2D(60, kernel_size=(4, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.025))
    model.add(Conv2D(90, kernel_size=(4, 2), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.05))
    model.add(Dense(8))

    inputs = Input(shape=(30, 2, 2))
    outputs = model(inputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])

    # training model
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # test model
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Loss : ', score[0])
    print('Accuracy : ', score[1])

if __name__ == '__main__':

    main()
