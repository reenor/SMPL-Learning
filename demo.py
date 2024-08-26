import os
import torch
from smplx import SMPL

### Configuration ###
SMPL_MODEL_DIR='D:/Projects/SMPL-Learning/data/models/smpl/removed_chumpy_objects/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
#####################


def main():
    batch_size = 1

    # Generate random pose and shape parameters
    global_orient = torch.zeros(batch_size, 3)
    pose_params = torch.rand(batch_size, 69) * 0.2
    shape_params = torch.rand(batch_size, 300) * 0.03
    
    smpl_object = SMPL(model_path=SMPL_MODEL_DIR,
                        body_pose=pose_params,
                        global_orient=global_orient,
                        betas=shape_params, num_betas=300)
    # smpl_object = SMPL(model_path=SMPL_MODEL_DIR)

    smpl_output = smpl_object.forward()
    joints = smpl_output.joints
    vertices = smpl_output.vertices
    print(joints)


if __name__ == '__main__':
    main()
