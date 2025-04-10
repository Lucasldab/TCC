import Sampling

class generateHyperparameters:
    def __init__(self, trainQuantity,samplingMethod,learningRate = (0.0001, 0.01),batchSize = (32., 128.), optimizer = (1., 9.)):
        self.sampleMethod = samplingMethod
        self.trainQuantity = trainQuantity

        if self.sampleMethod == 'grid':
            self.optimizer = Sampling.grid_sampling(self.trainQuantity, *optimizer)
            self.learningRate = Sampling.grid_sampling(self.trainQuantity, *learningRate,round=False)
            self.batchSize = Sampling.grid_sampling(self.trainQuantity, *batchSize)
        elif self.sampleMethod == 'LHS':
            self.optimizer = Sampling.lhs_sampling(self.trainQuantity, *optimizer)
            self.learningRate = Sampling.lhs_sampling(self.trainQuantity, *learningRate,round=False)
            self.batchSize = Sampling.lhs_sampling(self.trainQuantity, *batchSize)
        elif self.sampleMethod == 'random':
            self.optimizer = Sampling.random_sampling(self.trainQuantity, *optimizer)
            self.learningRate = Sampling.random_sampling(self.trainQuantity, *learningRate,round=False)
            self.batchSize = Sampling.random_sampling(self.trainQuantity, *batchSize)

    def generateFullyConnectedMLP(self,denseLayers):
        if self.sampleMethod == 'grid':
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.grid_sampling(self.trainQuantity, *denseLayers[i])
        elif self.sampleMethod == 'LHS':
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *denseLayers[i])
        elif self.sampleMethod == 'random':
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.random_sampling(self.trainQuantity, *denseLayers[i])

        return self.optimizer,self.learningRate,self.batchSize,denseLayers
    
    def generateCNN(self,convolutedLayers,denseLayers,filterLayers, maxPoolingLayers):
        if self.sampleMethod == 'grid':
            for i in range(0,len(convolutedLayers)):
                convolutedLayers[i] = Sampling.grid_sampling(self.trainQuantity, *convolutedLayers[i])
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.grid_sampling(self.trainQuantity, *denseLayers[i])
            for i in range(0,len(filterLayers)):
                filterLayers[i] = Sampling.grid_sampling(self.trainQuantity, *filterLayers[i])
                maxPoolingLayers[i] = Sampling.grid_sampling(self.trainQuantity, *maxPoolingLayers[i])
        elif self.sampleMethod == 'LHS':
            for i in range(0,len(convolutedLayers)):
                convolutedLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *convolutedLayers[i])
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *denseLayers[i])
            for i in range(0,len(filterLayers)):
                filterLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *filterLayers[i])
                maxPoolingLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *maxPoolingLayers[i])
        elif self.sampleMethod == 'random':
            for i in range(0,len(convolutedLayers)):
                convolutedLayers[i] = Sampling.random_sampling(self.trainQuantity, *convolutedLayers[i])
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.random_sampling(self.trainQuantity, *denseLayers[i])
            for i in range(0,len(filterLayers)):
                filterLayers[i] = Sampling.random_sampling(self.trainQuantity, *filterLayers[i])
                maxPoolingLayers[i] = Sampling.random_sampling(self.trainQuantity, *maxPoolingLayers[i])
        return self.optimizer,self.learningRate,self.batchSize,convolutedLayers,denseLayers,filterLayers,maxPoolingLayers
    
    def generateVGG16(self,convolutedLayers,denseLayers,filterLayers, maxPoolingLayers):
        if self.sampleMethod == 'grid':
            for i in range(0,len(convolutedLayers)):
                convolutedLayers[i] = Sampling.grid_sampling(self.trainQuantity, *convolutedLayers[i])
                filterLayers[i] = Sampling.grid_sampling(self.trainQuantity, *filterLayers[i])
            for i in range(0,len(maxPoolingLayers)):
                maxPoolingLayers[i] = Sampling.grid_sampling(self.trainQuantity, *maxPoolingLayers[i])
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.grid_sampling(self.trainQuantity, *denseLayers[i])
        elif self.sampleMethod == 'LHS':
            for i in range(0,len(convolutedLayers)):
                convolutedLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *convolutedLayers[i])
                filterLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *filterLayers[i])
            for i in range(0,len(maxPoolingLayers)):
                maxPoolingLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *maxPoolingLayers[i])
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.lhs_sampling(self.trainQuantity, *denseLayers[i])
        elif self.sampleMethod == 'random':
            for i in range(0,len(convolutedLayers)):
                convolutedLayers[i] = Sampling.random_sampling(self.trainQuantity, *convolutedLayers[i])
                filterLayers[i] = Sampling.random_sampling(self.trainQuantity, *filterLayers[i])
            for i in range(0,len(maxPoolingLayers)):
                maxPoolingLayers[i] = Sampling.random_sampling(self.trainQuantity, *maxPoolingLayers[i])
            for i in range(0,len(denseLayers)):
                denseLayers[i] = Sampling.random_sampling(self.trainQuantity, *denseLayers[i])
        return self.optimizer,self.learningRate,self.batchSize,convolutedLayers,denseLayers,filterLayers,maxPoolingLayers