import features as f
import data_io as d
import numpy as np

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

def extrair_tudo():
    combined = new_features1()

    print "Train"
    train = d.read_train_pairs()
    train_att = combined.fit_transform(train)
    np.save(train_att, open("train_att.npy", "wb"))


    print "Train1"
    valid = d.read_valid_pairs()
    valid_att = combined.fit_transform(valid)
    np.save(valid_att, open("valid_att.npy", "wb"))

    print "Train2"
    sup1 = d.read_sup1_train_pairs()
    sup1_att = combined.fit_transform(sup1)
    np.save(sup1_att, open("sup1_att.npy", "wb"))

    print "Train3"
    sup2 = d.read_sup2_train_pairs()
    sup2_att = combined.fit_transform(sup2)
    np.save(sup1_att, open("sup2_att.npy", "wb"))

    print "Train4"
    sup3 = d.read_sup3_train_pairs()
    sup3_att = combined.fit_transform(sup3)
    np.save(sup1_att, open("sup3_att.npy", "wb"))

if __name__ == "__main__":
    extrair_tudo()
