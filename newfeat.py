import feature_extractor as feat
import data_io as d
import numpy as np
import pickle

def ext():
    # Read the pairs
    print "Read Pairs"
    print "Read Train"
    train = d.read_train_pairs()
    print "Read Valid"
    valid = d.read_valid_pairs()
    print "Read Sup1"
    sup1 = d.read_sup1_train_pairs()
    print "Read Sup2"
    sup2 = d.read_sup2_train_pairs()
    print "Read Sup3"
    sup3 = d.read_sup3_train_pairs()

    # Get the feature extractor
    combined = feat.feature_extractor()

    # Extract the features
    print 'Extract the features'
    print "Extract Train"
    train_att = combined.fit_transform(train)
    print "Extract Valid"
    valid_att = combined.fit_transform(valid)
    print "Extract Sup1"
    sup1_att = combined.fit_transform(sup1)
    print "Extract Sup2"
    sup2_att = combined.fit_transform(sup2)
    print "Extract Sup3"
    sup3_att = combined.fit_transform(sup3)

    print "Join"
    total_new_att = np.vstack((train_att, valid_att, sup1_att, sup2_att, sup3_att))

    # Save extracted data
    total_new_att.tofile('total_new_att.np')
    pickle.dump(total_new_att.shape, open('total_new_att_shape.[', 'wb'))

    return total_new_att

if __name__ == "__main__":
    ext()

