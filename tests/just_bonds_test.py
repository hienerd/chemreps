import numpy as np
from chemreps.bagger import BagMaker
import chemreps.just_bonds as jb


def test_bag_maker():
    bags_true = {'CC': 7, 'HC': 10, 'OC': 2, 'OH': 2}
    bagger = BagMaker('JustBonds', 'data/sdf/')
    print(bagger.bags)
    assert bagger.bag_sizes == bags_true


def test_just_bonds():
    jbs_true = np.array([36.  , 36.  , 36.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  9.67,
        9.67,  9.67,  9.67,  9.67,  9.67,  9.67,  9.67,  9.67,  9.67,
        0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ], dtype=np.float16)
    bagger = BagMaker('JustBonds', 'data/sdf/')
    jbs = jb.bonds('data/sdf/butane.sdf', bagger.bags, bagger.bag_sizes)
    # print([jbs])
    assert np.allclose(jbs, jbs_true, 1e-4) == True


if __name__ == "__main__":
    print("This is a test of the bag of bonds representation in chemreps to be evaluated with pytest")
