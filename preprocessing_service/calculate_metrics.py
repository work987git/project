from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import datetime



def calculate_metrics_multi_label(run_id, ground_truth, prediction, flag):
    labels = ground_truth
    predicted_labels = prediction
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels, average='micro')
    recall = recall_score(labels, predicted_labels, average='micro')
    f1 = f1_score(labels, predicted_labels, average='micro')
    
    
    with mlflow.start_run(run_id=run_id):
        date = datetime.datetime.now().isoformat()
        tags = {
                "tag_key": f"pred_on_{flag}_{run_id}_{str(date)}"
            }
        mlflow.set_tags(tags)
        mlflow.log_metric(f"{flag}_accuracy", accuracy)
        mlflow.log_metric(f"{flag}_precision", precision)
        mlflow.log_metric(f"{flag}_recall", recall)
        mlflow.log_metric(f"{flag}_f1", f1)
    