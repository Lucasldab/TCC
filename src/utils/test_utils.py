import generateHyperparameters as gen
from optimizerSelector import numberToName
import csv
from os import path

# Define the number of CNN models to train
trainQT = 100000
method= 'grid'
dataSamples = 20
model = 'VGG16'
for samples in range(0,dataSamples):

    csvName = 'data/'+method+'/'+model+'_'+method+'_Hyperparameters_'+ str(samples+1) +'.csv'

    if not path.exists(csvName):
        with open(csvName, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Optimizer","Learning_Rate","Batch_Size","Convoluted_Layers1","Convoluted_Layers2","Convoluted_Layers3","Convoluted_Layers4","Convoluted_Layers5","Convoluted_Layers6","Convoluted_Layers7","Convoluted_Layers8","Convoluted_Layers9","Convoluted_Layers10","Convoluted_Layers11","Convoluted_Layers12","Convoluted_Layers13","Dense_Layers1","Dense_Layers2","Filter_Layers1","Filter_Layers2","Filter_Layers3","Filter_Layers4","Filter_Layers5","Filter_Layers6","Filter_Layers7","Filter_Layers8","Filter_Layers9","Filter_Layers10","Filter_Layers11","Filter_Layers12","Filter_Layers13","Max_Pooling_Layers1","Max_Pooling_Layers2","Max_Pooling_Layers3","Max_Pooling_Layers4","Max_Pooling_Layers5"])
            file.close()

    generator = gen.generateHyperparameters(trainQuantity=trainQT, samplingMethod=method,optimizer=(0., 8.))
    convLayers=[(32., 128.),(32., 128.),(64., 256.),(64., 256.),(128., 512.),(128., 512.),(128., 512.),(256., 1024.),(256., 1024.),(256., 1024.),(512., 2048.),(512., 2048.),(512., 2048.)]
    denLayers=[(2048., 8192.)]*2
    filLayers=[(1., 3.)]*len(convLayers)
    maxPLayers=[(1., 3.)]*5

    (optimizer,
     learningRate,
     batchSize,
     convolutedLayers,
     denseLayers,
     filterLayers,
     maxPoolingLayers) = generator.generateVGG16(convolutedLayers=convLayers,
                                                 denseLayers=denLayers,
                                                 filterLayers=filLayers,
                                                 maxPoolingLayers=maxPLayers)

    #print(learningRate[0])

    for training in optimizer:
        name = numberToName(training)        
        #print(name, learningRate[training], batchSize[training])
        with open(csvName, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name,
                             learningRate[training],
                             batchSize[training],
                             convolutedLayers[0][training],
                             convolutedLayers[1][training],
                             convolutedLayers[2][training],
                             convolutedLayers[3][training],
                             convolutedLayers[4][training],
                             convolutedLayers[5][training],
                             convolutedLayers[6][training],
                             convolutedLayers[7][training],
                             convolutedLayers[8][training],
                             convolutedLayers[9][training],
                             convolutedLayers[10][training],
                             convolutedLayers[11][training],
                             convolutedLayers[12][training],
                             denseLayers[0][training],
                             denseLayers[1][training],
                             filterLayers[0][training],
                             filterLayers[1][training],
                             filterLayers[2][training],
                             filterLayers[3][training],
                             filterLayers[4][training],
                             filterLayers[5][training],
                             filterLayers[6][training],
                             filterLayers[7][training],
                             filterLayers[8][training],
                             filterLayers[9][training],
                             filterLayers[10][training],
                             filterLayers[11][training],
                             filterLayers[12][training],
                             maxPoolingLayers[0][training],
                             maxPoolingLayers[1][training],
                             maxPoolingLayers[2][training],
                             maxPoolingLayers[3][training],
                             maxPoolingLayers[4][training]])
            file.close()

        #self.optimizer,self.learningRate,self.batchSize,convolutedLayers,denseLayers,filterLayers,maxPoolingLayers,