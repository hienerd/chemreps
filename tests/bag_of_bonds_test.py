import numpy as np
import pytest as pt
from chemreps.bagger import BagMaker
from chemreps.bag_of_bonds import bag_of_bonds


# def test_bag_maker():
#     bags_true = {'C': 7, 'CC': 21, 'CH': 42, 'H': 10, 'HH': 45, 'O': 2, 'OC': 14, 'OH': 12, 'OO': 1}
#     bagger = BagMaker('BoB', 'data/sdf/')
#     assert bagger.bag_sizes == bags_true


def test_bag_of_bonds():
    bags_true = {'C': 7, 'CC': 21, 'CH': 42, 'H': 10, 'HH': 45, 'O': 2, 'OC': 14, 'OH': 12, 'OO': 1}
    bobs_true = np.array([36.84  , 36.84  , 36.84  , 36.84  ,  0.    ,  0.    ,  0.    ,
        0.    , 36.    , 36.    , 36.    , 20.78  , 20.78  , 13.6   ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  9.67  ,  9.67  ,  9.67  ,  9.67  ,  9.67  ,
        9.67  ,  9.67  ,  9.67  ,  9.67  ,  9.67  ,  5.55  ,  5.55  ,
        5.55  ,  5.55  ,  5.098 ,  5.098 ,  5.098 ,  5.098 ,  4.24  ,
        4.24  ,  3.95  ,  3.945 ,  3.758 ,  3.758 ,  3.758 ,  3.758 ,
        3.703 ,  3.703 ,  2.842 ,  2.842 ,  2.748 ,  2.748 ,  2.621 ,
        2.62  ,  2.389 ,  2.389 ,  2.062 ,  2.062 ,  1.853 ,  1.853 ,
        0.    ,  0.    ,  0.    ,  0.5   ,  0.5   ,  0.5   ,  0.5   ,
        0.5   ,  0.5   ,  0.5   ,  0.5   ,  0.5   ,  0.5   ,  0.    ,
        1.255 ,  1.255 ,  1.141 ,  1.141 ,  1.141 ,  1.141 ,  1.12  ,
        1.12  ,  0.975 ,  0.975 ,  0.806 ,  0.806 ,  0.689 ,  0.658 ,
        0.658 ,  0.6133,  0.6133,  0.612 ,  0.612 ,  0.5923,  0.5923,
        0.559 ,  0.559 ,  0.549 ,  0.549 ,  0.5327,  0.5327,  0.518 ,
        0.518 ,  0.4531,  0.452 ,  0.452 ,  0.3958,  0.3784,  0.3784,
        0.378 ,  0.378 ,  0.3743,  0.3743,  0.3196,  0.3196,  0.306 ,
        0.2896,  0.2896,  0.2605,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    ], dtype=np.float16)
    bagger = BagMaker('BoB', 'data/sdf/')
    assert bagger.bag_sizes == bags_true

    bobs = bag_of_bonds('data/sdf/butane.sdf', bagger.bags, bagger.bag_sizes)
    assert np.allclose(bobs, bobs_true, 1e-4) == True


if __name__ == "__main__":
    print("This is a test of the bag of bonds representation in chemreps to be evaluated with pytest")
