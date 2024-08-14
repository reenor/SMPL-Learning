import os
import torch
from smplx import SMPL

### Configuration ###
SMPL_MODEL_DIR='D:\\Projects\\SMPL-Learning\\data\\model\\'
#####################


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    smpl_object = SMPL(model_path=SMPL_MODEL_DIR, gender='neutral')
    smpl_output = smpl_object.forward()
    joints = smpl_output.joints
    vertices = smpl_output.vertices
    print(joints)


if __name__ == '__main__':
    main()
