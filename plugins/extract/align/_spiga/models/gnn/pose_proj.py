import torch
import math


def euler_to_rotation_matrix(euler):
    # http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
    # Change coordinates system

    # Convert to radians
    yaw = -(euler[:, 0] - 90)
    pitch = -euler[:, 1]
    roll = -(euler[:, 2] + 90)
    rad = torch.stack((yaw, pitch, roll), dim=1) * (math.pi / 180.0)
    cy = torch.cos(rad[:, 0])
    sy = torch.sin(rad[:, 0])
    cp = torch.cos(rad[:, 1])
    sp = torch.sin(rad[:, 1])
    cr = torch.cos(rad[:, 2])
    sr = torch.sin(rad[:, 2])

    # Init R matrix tensors
    working_device = euler.device
    Ry = torch.zeros((euler.shape[0], 3, 3), device=working_device, dtype=euler.dtype)
    Rp = torch.zeros((euler.shape[0], 3, 3), device=working_device, dtype=euler.dtype)
    Rr = torch.zeros((euler.shape[0], 3, 3), device=working_device, dtype=euler.dtype)

    # Yaw
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy

    # Pitch
    Rp[:, 0, 0] = cp
    Rp[:, 0, 1] = -sp
    Rp[:, 1, 0] = sp
    Rp[:, 1, 1] = cp
    Rp[:, 2, 2] = 1.0

    # Roll
    Rr[:, 0, 0] = 1.0
    Rr[:, 1, 1] = cr
    Rr[:, 1, 2] = -sr
    Rr[:, 2, 1] = sr
    Rr[:, 2, 2] = cr

    return torch.matmul(torch.matmul(Ry, Rp), Rr)


def projectPoints(pts, rot, trl, cam_matrix):

    # Get working device
    working_device = pts.device

    # Perspective projection model
    trl = trl.unsqueeze(2)
    extrinsics = torch.cat((rot, trl), 2)
    proj_matrix = torch.matmul(cam_matrix, extrinsics)

    # Homogeneous landmarks
    ones = torch.ones(
        pts.shape[:2],
        device=working_device,
        dtype=pts.dtype,
        requires_grad=trl.requires_grad,
    )
    ones = ones.unsqueeze(2)
    pts_hom = torch.cat((pts, ones), 2)

    # Project landmarks
    pts_proj = pts_hom.permute((0, 2, 1))  # Transpose
    pts_proj = torch.matmul(proj_matrix, pts_proj)
    pts_proj = pts_proj.permute((0, 2, 1))
    pts_proj = pts_proj / pts_proj[:, :, 2].unsqueeze(2)  # Lambda = 1

    return pts_proj[:, :, :-1]
