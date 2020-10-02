import kiwi

train = '/home/snchen/project/openkiwi-init-addsent/experiments/train_predictor.yaml'
kiwi.train(train)
train2 = '/home/snchen/project/openkiwi-init-addsent/experiments/train_estimator.yaml'
kiwi.train(train2)

predict = '/home/snchen/project/openkiwi-init-addsent/experiments/predict_estimator.yaml'
kiwi.predict(predict)
evaluate = '/home/snchen/project/openkiwi-init-addsent/experiments/evaluate_estimator.yaml'
kiwi.evaluate(evaluate)
