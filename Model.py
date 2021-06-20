from sklearn.svm import SVC

def Model(x, y, a):
	model_svc=SVC(C=100,kernel='rbf')
	model_svc.fit(x, y)
	y_predicted=model_svc.predict(a)
	
	return y_predicted
