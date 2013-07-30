import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import score as s

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)), # 0
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)), #1
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)), #2
                ('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)), #3
                ('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)), #4
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)), #5
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)), #6
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference)), #7

				('IGCI11',['A','B'], f.MultiColumnTransform(transformer=f.igci11)), #8
				('IGCI12',['A','B'], f.MultiColumnTransform(transformer=f.igci12)), #9
				('IGCI21',['A','B'], f.MultiColumnTransform(transformer=f.igci21)), #10
				('IGCI22',['A','B'], f.MultiColumnTransform(transformer=f.igci22)), #11

				('N_IGCI11',['A','B'], f.MultiColumnTransform(transformer=f.n_igci11)), #8 #12 (#0 * #8)
				('N_IGCI12',['A','B'], f.MultiColumnTransform(transformer=f.n_igci12)), #9 #13
				('N_IGCI21',['A','B'], f.MultiColumnTransform(transformer=f.n_igci21)), #10 #14
				('N_IGCI22',['A','B'], f.MultiColumnTransform(transformer=f.n_igci22)),  #11 #15

				('Mean A', 'A', f.SimpleTransform(transformer=f.media)), #16
				('Mean B', 'B', f.SimpleTransform(transformer=f.media)), #17
				('Mediana A', 'A', f.SimpleTransform(transformer=f.mediana)), #18
				('Mediana B', 'B', f.SimpleTransform(transformer=f.mediana)), #19
				('Intervalo A', 'A', f.SimpleTransform(transformer=f.intervalo)), #20
				('Intervalo B', 'B', f.SimpleTransform(transformer=f.intervalo)), #21
				('Desvio Padrao A', 'A', f.SimpleTransform(transformer=f.desvio_padrao)), #22
				('Desvio Padrao B', 'B', f.SimpleTransform(transformer=f.desvio_padrao))] #23
    combined = f.FeatureMapper(features)
    return combined

def new_features1():
    features = [
                ('A: Random Forest Score', ['A','B'], f.MultiColumnTransform(transformer=f.RFscore_one)),
                ('B: Random Forest Score', ['B','A'], f.MultiColumnTransform(transformer=f.RFscore_one)),
                ('AB: Slope Entropy', ['A','B'], f.MultiColumnTransform(transformer=f.slope_entropy)),
                ('BA: Slope Entropy', ['B','A'], f.MultiColumnTransform(transformer=f.slope_entropy_rev)),
                ('Cagegoric Prob Indep', ['A','B'], f.MultiColumnTransform(transformer=f.categoric)),
                ('Cagegoric Cond Prob', ['A','B'], f.MultiColumnTransform(transformer=f.categoric_cond_prob)),
                ('Cagegoric Cond Prob', ['A','B'], f.MultiColumnTransform(transformer=f.categoric_cond_prob)),
                ('Slope Entropy Difference', ['A','B'], f.MultiColumnTransform(transformer=f.slope_entropy_diff)),
                ('LinRegres', ['A','B'], f.MultiColumnTransform(transformer=f.linregress)),
                ('QuadRegres', ['A','B'], f.MultiColumnTransform(transformer=f.quadregress)),
                ('PolyRegres', ['A','B'], f.MultiColumnTransform(transformer=f.polyregress)),
                ('SineRegres', ['A','B'], f.MultiColumnTransform(transformer=f.sineregress)),
                ('Difference Correlation', ['A','B'], f.MultiColumnTransform(transformer=f.diffcor))
                ]
    combined = f.FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=250,
                                                verbose=2,
                                                n_jobs=-1,
                                                min_samples_split=10,
                                                random_state=1))]
    return Pipeline(steps)

def getdata():
	total = np.fromfile('extracted/total.np')
	total = total.reshape(total.shape[0] / 24, 24)
	train_traget = data_io.read_train_target()
	sup1_traget = data_io.read_sup1_train_target()
	sup2_traget = data_io.read_sup2_train_target()
	sup3_traget = data_io.read_sup3_train_target()
	total_target = np.hstack((train_traget.Target, sup1_traget.Target, sup2_traget.Target, sup3_traget.Target))
	return total, total_target

def main():
	print("Loading train data")
	total, total_target = getdata()

	print("Normalizing data")
	mu = np.mean(total, axis=0)
	sigma = np.mean(total, axis=0)
	X_norm = (total - mu) / sigma

	print("PCA")
	pca = PCA(n_components=16, whiten=True)
	pca.fit(X_norm)
	X_pca = pca.transform(X_norm)

	print("Split train")
	X_train, X_test, y_train, y_test = train_test_split(
		X_pca, total_target, test_size=0.25, random_state=0)
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
		'C': [1, 10, 100, 1000]},
		{'kernel':['linear'], 'C': [1, 10, 100, 1000]}]

	print("GridSearchCV")
	clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3, score_func=s.bidirectional_auc, n_jobs=-1, verbose=3)
	clf.fit(X_train, y_train)

	print("Best parameters")
	print(clf.best_estimator_)

	for params, mean_score, scores in clf.cv_scores_:
		print("%0.3f (+/-$0.03f) for %r"
			% (mean_score, scores.std() / 2, params))

	y_true, y_pred = y_test, clf.predit(X_test)
	print(classification_report(y_true, y_pred))
	print(s.bidirectional_auc(y_true, y_pred))

	print("Saving the classifier")
	data_io.save_model(clf)

if __name__=="__main__":
    main()
