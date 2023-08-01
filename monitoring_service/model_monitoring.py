import argparse
import mlflow
import sys

sys.path.append(r'C:\Users\U1153226\Documents\pb_pivot2\mlflow\clinical_trial_repo\clinical_project-main\clinical_prj\project')

from feedback_service import feedback
from training_class import DataProcessor

def model_monitor(run_id, config_path):
    
    mlflow.set_tracking_uri("http://localhost:5000") #TODO read from config
    run = mlflow.get_run(run_id)

    # call feedback service
    feedback_csv_file_path = feedback.feedback_service(run_id)
    training_cls = DataProcessor(config_path,feedback_csv_file_path )
    print('PREDICTIONS ON FEEDBACK DONE')

    # compare feedback and testing metrics to detect model drift
    if (run.data.metrics['test_f1'] > run.data.metrics['feedback_f1'] 
        or run.data.metrics['test_accuracy'] > run.data.metrics['feedback_accuracy'] 
        or run.data.metrics['test_precision'] > run.data.metrics['feedback_precision']):
        print('MODEL DRIFT DETECTED')
        training_cls.run_retraining()

    else:
        training_cls.run_retraining()
        print('NO MODEL DRIFT DETECTED')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    model_monitor(args.run_id, args.config_path)
