import feature_extractor as feat
import data_io as d
import numpy as np

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

    # Save the extracted features
    print "Save the features"
    print "Save Train"
    train_att.tofile('extracted/train_att_1.np')
    print "Save Valid"
    valid_att.tofile('extracted/valid_att_1.np')
    print "Save Sup1"
    sup1_att.tofile('extracted/sup1_att_1.np')
    print "Save Sup2"
    sup2_att.tofile('extracted/sup2_att_1.np')
    print "Save Sup3"
    sup3_att.tofile('extracted/sup3_att_1.np')

    print "Join"
    np.vstack((train_att, valid_att, sup1_att, sup2_att, sup3_att))

    print "Fim"
    return train_att, valid_att, sup1_att, sup2_att, sup3_att, joined_att

