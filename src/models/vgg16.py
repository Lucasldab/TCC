import pandas as pd
import neuralNetwork
import tensorflow.keras.datasets.cifar100 as cifar100 # type: ignore
import tensorflow.keras.backend as backend # type: ignore
import csv
import os

trains = 1000
trainModel = 'VGG16'
samplingMethod = 'LHS'#input("Whitch sampling method?")
startingDataset = 0#int(input("Whitch sampling number?"))

for datasetNumber in range(startingDataset,20):

    datasetLocal = 'data/'+ samplingMethod +'/'+trainModel+'_'+ samplingMethod +'_Hyperparameters_'+ str(datasetNumber+1) +'.csv'
    trainingFile = 'trainings/'+trainModel+'_'+ samplingMethod +'/training_'+ str(datasetNumber+1) +'.csv'

    print('Dataset: '+trainModel+'_'+ samplingMethod +'_Hyperparameters_'+ str(datasetNumber+1) +'.csv')
    print('Training: training_'+ str(datasetNumber+1) +'.csv')

    # Check if the folder exists
    if not os.path.exists('trainings/'+trainModel+'_'+ samplingMethod):
        # Create the folder
        os.makedirs('trainings/'+trainModel+'_'+ samplingMethod)
    if not os.path.exists(trainingFile):
        with open(trainingFile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Optimizer","Learning_Rate","Batch_Size","Convoluted_Layers1","Convoluted_Layers2","Convoluted_Layers3","Convoluted_Layers4","Convoluted_Layers5","Convoluted_Layers6","Convoluted_Layers7","Convoluted_Layers8","Convoluted_Layers9","Convoluted_Layers10","Convoluted_Layers11","Convoluted_Layers12","Convoluted_Layers13","Dense_Layers1","Dense_Layers2","Filter_Layers1","Filter_Layers2","Filter_Layers3","Filter_Layers4","Filter_Layers5","Filter_Layers6","Filter_Layers7","Filter_Layers8","Filter_Layers9","Filter_Layers10","Filter_Layers11","Filter_Layers12","Filter_Layers13","Max_Pooling_Layers1","Max_Pooling_Layers2","Max_Pooling_Layers3","Max_Pooling_Layers4","Max_Pooling_Layers5,Loss"])
            file.close()

    data = pd.read_csv(datasetLocal)
    optimizer_name = data["Optimizer"].values
    learning_rate = data["Learning_Rate"].values
    batch_size = data["Batch_Size"].values
    conv_n1 = data["Convoluted_Layers1"].values
    conv_n2 = data["Convoluted_Layers2"].values
    conv_n3 = data["Convoluted_Layers3"].values
    conv_n4 = data["Convoluted_Layers4"].values
    conv_n5 = data["Convoluted_Layers5"].values
    conv_n6 = data["Convoluted_Layers6"].values
    conv_n7 = data["Convoluted_Layers7"].values
    conv_n8 = data["Convoluted_Layers8"].values
    conv_n9 = data["Convoluted_Layers9"].values
    conv_n10 = data["Convoluted_Layers10"].values
    conv_n11 = data["Convoluted_Layers11"].values
    conv_n12 = data["Convoluted_Layers12"].values
    conv_n13 = data["Convoluted_Layers13"].values
    dense_n1 = data["Dense_Layers1"].values
    dense_n2 = data["Dense_Layers2"].values
    filter_n1 = data["Filter_Layers1"].values
    filter_n2 = data["Filter_Layers2"].values
    filter_n3 = data["Filter_Layers3"].values
    filter_n4 = data["Filter_Layers4"].values
    filter_n5 = data["Filter_Layers5"].values
    filter_n6 = data["Filter_Layers6"].values
    filter_n7 = data["Filter_Layers7"].values
    filter_n8 = data["Filter_Layers8"].values
    filter_n9 = data["Filter_Layers9"].values
    filter_n10 = data["Filter_Layers10"].values
    filter_n11 = data["Filter_Layers11"].values
    filter_n12 = data["Filter_Layers12"].values
    filter_n13 = data["Filter_Layers13"].values
    maxP_n1 = data["Max_Pooling_Layers1"].values
    maxP_n2 = data["Max_Pooling_Layers2"].values
    maxP_n3 = data["Max_Pooling_Layers3"].values
    maxP_n4 = data["Max_Pooling_Layers4"].values
    maxP_n5 = data["Max_Pooling_Layers5"].values
    for training in range(0, trains):
        print("Training: ", training+1)
        print("Dataset: ", samplingMethod + ' ' + str(datasetNumber+1))

        ConvolutedLayers = [conv_n1[training], conv_n2[training], conv_n3[training], conv_n4[training], conv_n5[training], conv_n6[training], conv_n7[training], conv_n8[training], conv_n9[training], conv_n10[training], conv_n11[training], conv_n12[training], conv_n13[training]]
        denseLayers = [dense_n1[training], dense_n2[training]]
        filterLayers = [filter_n1[training], filter_n2[training], filter_n3[training], filter_n4[training], filter_n5[training], filter_n6[training], filter_n7[training], filter_n8[training], filter_n9[training], filter_n10[training], filter_n11[training], filter_n12[training], filter_n13[training]]
        maxPoolingLayers = [maxP_n1[training], maxP_n2[training], maxP_n3[training], maxP_n4[training], maxP_n5[training]]

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        NN = neuralNetwork.NeuralNetwork(optimizer=optimizer_name[training],learningRate=learning_rate[training])
        model = NN.createVGG16Model(ConvolutedLayers=ConvolutedLayers,denseLayers=denseLayers,filterLayers=filterLayers,maxPoolingLayers=maxPoolingLayers)

        # Train the model
        history = model.fit(x_train, y_train, epochs=5, batch_size=batch_size[training])
        # Extract the loss from the training history
        history_dict = history.history
        final_loss = history_dict['loss'][5-1]
        print(final_loss)
        
        with open(trainingFile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([optimizer_name[training],
                             learning_rate[training],
                             batch_size[training],
                             ConvolutedLayers[0],
                             ConvolutedLayers[1],
                             ConvolutedLayers[2],
                             ConvolutedLayers[3],
                             ConvolutedLayers[4],
                             ConvolutedLayers[5],
                             ConvolutedLayers[6],
                             ConvolutedLayers[7],
                             ConvolutedLayers[8],
                             ConvolutedLayers[9],
                             ConvolutedLayers[10],
                             ConvolutedLayers[11],
                             ConvolutedLayers[12],
                             denseLayers[0],
                             denseLayers[1],
                             filterLayers[0],
                             filterLayers[1],
                             filterLayers[2],
                             filterLayers[3],
                             filterLayers[4],
                             filterLayers[5],
                             filterLayers[6],
                             filterLayers[7],
                             filterLayers[8],
                             filterLayers[9],
                             filterLayers[10],
                             filterLayers[11],
                             filterLayers[12],
                             maxPoolingLayers[0],
                             maxPoolingLayers[1],
                             maxPoolingLayers[2],
                             maxPoolingLayers[3],
                             maxPoolingLayers[4]],
                             final_loss)
            file.close()

        backend.clear_session()
