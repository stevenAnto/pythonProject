import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tsf

def function():
    inputs  = np.array([1, 6,30,7,70,43,503,201,1005,99], dtype=float)
    outputs = np.array([0.0254,0.1524,0.762,0.1778,0.1778,1.0922,12.776,5.1054,25.527,2.514],dtype=float)
    print(inputs)

    #topografia de la red
    #Una capa de entrada y una capa de salida
    capa1 = tsf.keras.layers.Dense(units=1,input_shape=[1]) #todas con todos entre capas
    #Se dfine el tipo de red
    modelo = tsf.keras.Sequential([capa1])

    #Asignamos optimizador y metricas de perdida
    #El optimizador va reduciendo los errores
    modelo.compile(
        optimizer= tsf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    print('Se esta entrenando la red')

    #entrenando el modelo
    #epoch son los ciclos de enrenamientos
    #verbose para visivilizar los ciclos de entrenamiento
    entrenamiento = modelo.fit(inputs,outputs,epochs=500,verbose=False)

    #Guardar la red
    modelo.save('RedNeuronal.h5')
    modelo.save_weights('pesos.h5')

    plt.xlabel('ciclos de entrenamiento')
    plt.ylabel('errores')
    plt.plot(entrenamiento.history['loss'])
    plt.show()

    #verificar que la red se entreno
    print('Terminado')



    #prediccion
    i=input("ingrese un valor en pulgadas")
    i=float(i)

    prediccion =modelo.predict([i])
    print("El valor en metros :",str(prediccion))

if __name__ == '__main__':

    function()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
