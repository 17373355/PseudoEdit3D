SMPLH_NUM_JOINTS = 52
SMPLH_POSE_DIM = 156
SMPLH_HAND_START = 22

SMPLH_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Index1", "L_Index2", "L_Index3", "L_Middle1",
    "L_Middle2", "L_Middle3", "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1",
    "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1",
    "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1",
    "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1",
    "R_Thumb2", "R_Thumb3",
]

JOINT_INDEX = {name: idx for idx, name in enumerate(SMPLH_JOINT_NAMES)}

BODY_PART_TO_JOINTS = {
    "left_arm": [JOINT_INDEX["L_Collar"], JOINT_INDEX["L_Shoulder"], JOINT_INDEX["L_Elbow"], JOINT_INDEX["L_Wrist"]],
    "right_arm": [JOINT_INDEX["R_Collar"], JOINT_INDEX["R_Shoulder"], JOINT_INDEX["R_Elbow"], JOINT_INDEX["R_Wrist"]],
    "both_arms": [
        JOINT_INDEX["L_Collar"], JOINT_INDEX["L_Shoulder"], JOINT_INDEX["L_Elbow"], JOINT_INDEX["L_Wrist"],
        JOINT_INDEX["R_Collar"], JOINT_INDEX["R_Shoulder"], JOINT_INDEX["R_Elbow"], JOINT_INDEX["R_Wrist"],
    ],
    "torso": [JOINT_INDEX["Pelvis"], JOINT_INDEX["Spine1"], JOINT_INDEX["Spine2"], JOINT_INDEX["Spine3"], JOINT_INDEX["Neck"]],
    "whole_body": list(range(22)),
}
