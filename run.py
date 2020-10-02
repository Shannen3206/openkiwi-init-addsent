import kiwi

train = '/home/snchen/project/openkiwi-init-addsent/experiments/train_nuqe.yaml'
kiwi.train(train)

predict = '/home/snchen/project/openkiwi-init-addsent/experiments/predict_nuqe.yaml'
kiwi.predict(predict)
evaluate = '/home/snchen/project/openkiwi-init-addsent/experiments/evaluate_nuqe.yaml'
kiwi.evaluate(evaluate)
