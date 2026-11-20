from pathology_anomaly.experiment import evaluate,make_data
def test_shapes():
 x,y=make_data(20,2);assert x.shape==(20,36) and y.shape==(20,)
def test_scores():
 r=evaluate(2);assert 0<=r["auroc"]<=1 and 0<=r["average_precision"]<=1

