import dataTreatment
import GaussianRegression

#Data treatment
data = pd.read_csv('data/training_CNN_results_v3.csv')
clean_data = dataTreatment.clean_data(data)
half_data,other_half_data = dataTreatment.divide_samplings(clean_data)
loss_data,data_only,smallest_loss_local = dataTreatment.data_from_loss(half_data)


#Gaussian Process
y_mean,y_cov = GaussianRegression.gaussianProcess(data_only,loss_data,other_half_data)

#Acquisition Function
surrogate_values = SurrogateFunction.expected_improvement(y_mean,y_cov,loss_data)