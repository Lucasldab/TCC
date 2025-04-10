# Move model files
Move-Item -Path "MLP.py" -Destination "src\models\"
Move-Item -Path "CNN training.py" -Destination "src\models\cnn.py"
Move-Item -Path "VGG16 training.py" -Destination "src\models\vgg16.py"
Move-Item -Path "neuralNetwork.py" -Destination "src\models\"

# Move optimizer files
Move-Item -Path "ParticleSwarmOptimization.py" -Destination "src\optimizers\pso.py"
Move-Item -Path "GRPSO.py" -Destination "src\optimizers\grpso.py"
Move-Item -Path "TPE MLP.py" -Destination "src\optimizers\tpe_mlp.py"
Move-Item -Path "TPE VGG16.py" -Destination "src\optimizers\tpe_vgg16.py"
Move-Item -Path "TPE CNN.py" -Destination "src\optimizers\tpe_cnn.py"
Move-Item -Path "GaussianRegression.py" -Destination "src\optimizers\gaussian_regression.py"
Move-Item -Path "GP.py" -Destination "src\optimizers\"
Move-Item -Path "CRU.py" -Destination "src\optimizers\"
Move-Item -Path "DDE.py" -Destination "src\optimizers\"
Move-Item -Path "optimizerSelector.py" -Destination "src\optimizers\"
Move-Item -Path "GRPSO MLP.py" -Destination "src\optimizers\grpso_mlp.py"
Move-Item -Path "PSO_CNN_Results.py" -Destination "src\optimizers\pso_cnn_results.py"
Move-Item -Path "PSO_MLP_Results.py" -Destination "src\optimizers\pso_mlp_results.py"

# Move utility files
Move-Item -Path "dataTreatment.py" -Destination "src\utils\"
Move-Item -Path "counting.py" -Destination "src\utils\"
Move-Item -Path "Sampling.py" -Destination "src\utils\"
Move-Item -Path "generateHyperparameters.py" -Destination "src\utils\"
Move-Item -Path "test.py" -Destination "src\utils\test_utils.py"

# Move notebooks
Move-Item -Path "*.ipynb" -Destination "notebooks\"

# Move data files
Move-Item -Path "data\*" -Destination "src\data\"

# Create necessary directories if they don't exist
New-Item -ItemType Directory -Force -Path "src\data\trainings"

# Move remaining files
Move-Item -Path "test.py" -Destination "src\utils\test_utils.py" -ErrorAction SilentlyContinue
Move-Item -Path "GRPSO MLP.py" -Destination "src\optimizers\grpso_mlp.py" -ErrorAction SilentlyContinue
Move-Item -Path "PSO_CNN_Results.py" -Destination "src\optimizers\pso_cnn_results.py" -ErrorAction SilentlyContinue
Move-Item -Path "PSO_MLP_Results.py" -Destination "src\optimizers\pso_mlp_results.py" -ErrorAction SilentlyContinue

# Move trainings directory
if (Test-Path "trainings") {
    Move-Item -Path "trainings\*" -Destination "src\data\trainings\" -Force
    Remove-Item -Path "trainings" -Force
}

# Clean up __pycache__ directories
Get-ChildItem -Path "." -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Move main script
Move-Item -Path "main.py" -Destination "src\" 